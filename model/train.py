import itertools

import torch
from torch import nn, optim
from tqdm import tqdm
import copy

from model import device
from model.models import reconstruct_Decoder, Encoder, end2end_Decoder, end2end_Decoder_ablation


def train_phase1(clean_dataloader, attribute_dims, n_epochs=10, lr=0.0002, b1=0.5, b2=0.999, seed=None, enc_hidden_dim = 64, encoder_num_layers = 4, decoder_num_layers=2, dec_hidden_dim = 64,
    ):
    '''
    :param clean_dataloader: only contains clean dataset
    :param attribute_dims:  Number of attribute values per attribute : list
    :param n_epochs:  number of epochs of training
    :param lr: adam: learning rate
    :param b1: adam: decay of first order momentum of gradient
    :param b2: adam: decay of first order momentum of gradient
    :param seed: value of Pytorch random seed
    :param enc_hidden_dim: encoder hidden dimensions :GRU
    :param encoder_num_layers: Number of encoder layers :GRU
    :param decoder_num_layers： Number of decoder layers :GRU
    :param dec_hidden_dim:  decoder hidden dimensions :GRU
    :return: encoder,decoder
    '''

    if type(seed) is int:
        torch.manual_seed(seed)

    encoder = Encoder(attribute_dims, enc_hidden_dim, encoder_num_layers ,dec_hidden_dim)
    decoder = reconstruct_Decoder(attribute_dims, enc_hidden_dim,decoder_num_layers ,dec_hidden_dim)

    encoder.to(device)
    decoder.to(device)

    optimizer = torch.optim.Adam(itertools.chain (encoder.parameters(), decoder.parameters()),lr=lr, betas=(b1, b2))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(n_epochs/2), gamma=0.1)

    print("*"*10+"training_1"+"*"*10)
    for epoch in range(int(n_epochs)):
        train_loss = 0.0
        train_num = 0
        for i, Xs in enumerate(tqdm(clean_dataloader)):
            mask = Xs[-1]
            Xs = Xs[:-1]
            mask=mask.to(device)
            for k ,X in enumerate(Xs):
                Xs[k]=X.to(device)

            s,enc_output = encoder(Xs)
            reconstruct_X = decoder(Xs,s,enc_output,mask)

            optimizer.zero_grad()

            loss=0.0
            for ij in range(len(attribute_dims)):
                #--------------
                # 除了每一个属性的起始字符之外,其他重建误差
                #---------------
                # pred=reconstruct_X[ij][:,1:,:].flatten(0,-2)
                pred = torch.softmax(reconstruct_X[ij][:, 1:, :], dim=2).flatten(0, -2)
                true=Xs[ij][:,1:].flatten()
                corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(device).reshape(-1,
                                                                                          reconstruct_X[0].shape[1] - 1)

                cross_entropys = -torch.log(corr_pred)
                loss += cross_entropys.masked_select((~mask[:, 1:])).mean()

            train_loss += loss.item() * Xs[0].shape[0]
            train_num +=Xs[0].shape[0]
            loss.backward()
            optimizer.step()
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_epoch=train_loss / train_num
        print(f"[Epoch {epoch+1:{len(str(n_epochs))}}/{n_epochs}] "
              f"[loss: {train_loss_epoch:3f}]")
        scheduler.step()

    return encoder,decoder



def train_phase2(dataloader, attribute_dims, max_len, encoder, decoder, p_lambda=1, n_epochs=10, lr=0.0002, b1=0.5, b2=0.999, seed=None, dec_hidden_dim = 64):
    '''
    :param dataloader: 平衡类别标签后的
    :param attribute_dims:  Number of attribute values per attribute : list
    :param max_len: max length of traces
    :param encoder: encoder of feature encoder
    :param decoder: decoder of feature encoder
    :param n_epochs:  number of epochs of training
    :param lr: adam: learning rate
    :param b1: adam: decay of first order momentum of gradient
    :param b2: adam: decay of first order momentum of gradient
    :param seed: value of Pytorch random seed
    :param dec_hidden_dim:  decoder hidden dimensions :GRU
    :return: end2end_decoder
    '''
    if type(seed) is int:
        torch.manual_seed(seed)

    end2end_decoder=end2end_Decoder(len(attribute_dims)*dec_hidden_dim+len(attribute_dims)*(max_len))
    end2end_decoder.to(device)
    optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), end2end_decoder.parameters(),decoder.parameters()), lr=lr,
                                     betas=(b1, b2))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(n_epochs/2), gamma=0.1)
    print("*" * 10 + "training_2" + "*" * 10)
    for epoch in range(int(n_epochs)):
        train_loss = 0.0
        train_num = 0
        for i, Xs in enumerate(tqdm(dataloader)):
            labels= Xs[-1]
            mask = Xs[-2]
            Xs = Xs[:-2]
            for k, X in enumerate(Xs):
                Xs[k] = X.to(device)
            labels=labels.to(device)
            mask = mask.to(device)
            s, enc_output = encoder(Xs)
            reconstruct_X = decoder(Xs,s,enc_output,mask)
            temp1 = []
            temp2=[]
            temp3 = []
            e = torch.zeros(len(labels)).to(device)
            for ij in range(len(attribute_dims)):
                pred = torch.softmax(reconstruct_X[ij][:, 1:, :],dim=2).flatten(0, -2)
                true = Xs[ij][:, 1:].flatten()
                # probs.append(pred.gather(1,true.view(-1, 1)).flatten().to(device).reshape((-1,reconstruct_X[0].shape[1]-1)))
                corr_pred = pred.gather(1,true.view(-1, 1)).flatten().to(device).reshape(-1,reconstruct_X[0].shape[1]-1)*(~mask[:, 1:])
                corr_pred[corr_pred==0]=1
                cross_entropys = -torch.log(corr_pred)

                cross_entropy_max=cross_entropys.max(1).values.unsqueeze(1)  ##最大的损失
                corr_pred_min = corr_pred.min(1).values.unsqueeze(1)
                e += cross_entropys.sum(1)/(~mask[:, 1:]).sum(1)
                temp1.append(cross_entropys)
                temp2.append(cross_entropy_max)
                temp3.append(corr_pred_min)
            temp1 = torch.cat(temp1,1)
            temp2 = torch.cat(temp2, 1)
            temp3 = torch.cat(temp3, 1)

            trace_level_abnormal_scores = end2end_decoder(torch.cat((torch.cat(s,1), temp1,temp2),1)).squeeze()

            optimizer.zero_grad()


            loss = torch.mean((1 - labels) * trace_level_abnormal_scores + labels * torch.pow(
                torch.log(trace_level_abnormal_scores), 2) + p_lambda * ((1 - labels) * e + labels * temp3.min(1).values))



            train_loss += loss.item() * Xs[0].shape[0]
            train_num += Xs[0].shape[0]
            loss.backward()
            optimizer.step()
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_epoch = train_loss / train_num
        print(f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{int(n_epochs)}] "
              f"[loss: {train_loss_epoch:3f}]")
        scheduler.step()

    return end2end_decoder



def train_phase3(dataloader, attribute_dims, encoder, decoder, n_epochs=5, lr=0.0002, b1=0.5, b2=0.999, seed=None ):
    '''
    :param dataloader: original dataset
    :param attribute_dims:  Number of attribute values per attribute : list
    :param encoder: encoder of feature encoder
    :param decoder: decoder of feature encoder
    :param n_epochs:  number of epochs of training
    :param lr: adam: learning rate
    :param b1: adam: decay of first order momentum of gradient
    :param b2: adam: decay of first order momentum of gradient
    :param seed: value of Pytorch random seed
    :return: reconstruct_encoder,reconstruct_decoder
    '''

    if type(seed) is int:
        torch.manual_seed(seed)

    reconstruct_encoder = copy.deepcopy(encoder)
    reconstruct_decoder = copy.deepcopy(decoder)

    optimizer = torch.optim.Adam(itertools.chain(reconstruct_encoder.parameters(), reconstruct_decoder.parameters()),lr=lr, betas=(b1, b2))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(n_epochs/2), gamma=0.1)
    print("*" * 10 + "training_3" + "*" * 10)
    for epoch in range(int(n_epochs)):
        train_loss = 0.0
        train_num = 0
        for i, Xs in enumerate(tqdm(dataloader)):
            labels = Xs[-1]
            mask = Xs[-2]
            Xs = Xs[:-2]
            mask = mask.to(device)
            labels = labels.to(device)
            for k, X in enumerate(Xs):
                Xs[k] = X.to(device)

            s, enc_output = reconstruct_encoder(Xs)
            reconstruct_X = reconstruct_decoder(Xs, s, enc_output,mask)

            optimizer.zero_grad()


            loss = 0.0
            temp = []
            for ij in range(len(attribute_dims)):
                # pred=reconstruct_X[ij][:,1:,:].flatten(0,-2)
                pred = torch.softmax(reconstruct_X[ij][:, 1:, :], dim=2).flatten(0, -2)
                true = Xs[ij][:, 1:].flatten()

                corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(device).reshape(-1, reconstruct_X[0].shape[
                    1] - 1) * (~mask[:, 1:])
                corr_pred[corr_pred == 0] = 1
                cross_entropys = -torch.log(corr_pred)

                corr_pred_min = corr_pred.min(1).values.unsqueeze(1) ##每一个轨迹当前属性的最小的概率
                cross_entropy_loss = cross_entropys.sum(1) / (~mask[:, 1:]).sum(1) ##每一个轨迹当前属性的交叉熵损失

                # loss +=  ((1-labels) *cross_entropy_loss).mean()

                loss += (1-labels) * cross_entropy_loss
                temp.append(corr_pred_min)
            temp = torch.cat(temp, 1)
            loss =(loss + 2 * (labels * temp.min(1).values)).mean()


            train_loss += loss.item() * Xs[0].shape[0]
            train_num += Xs[0].shape[0]
            loss.backward()
            optimizer.step()
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_epoch = train_loss / train_num
        print(f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{int(n_epochs)}] "
              f"[loss: {train_loss_epoch:3f}]")
        scheduler.step()

    return reconstruct_encoder,reconstruct_decoder

