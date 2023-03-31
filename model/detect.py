import numpy as np
import torch
from tqdm import tqdm

from model import device


def detect(encoder,decoder,end2end_decoder,reconstruct_encoder,reconstruct_decoder, dataloader, attribute_dims, attr_Shape):
    encoder.eval()
    decoder.eval()
    end2end_decoder.eval()
    reconstruct_encoder.eval()
    reconstruct_decoder.eval()

    pos=0
    with torch.no_grad():
        trace_level_abnormal_scores=[]
        attr_level_abnormal_scores=np.zeros(attr_Shape)
        print("*" * 10 + "detecting" + "*" * 10)
        for Xs in tqdm(dataloader):
            mask_c = Xs[-1]
            Xs = Xs[:-1]
            for k,X in enumerate(Xs):
                Xs[k] = X.to(device)
            mask=mask_c.to(device)

            s, enc_output = encoder(Xs)
            temp_X = decoder(Xs,  s, enc_output)

            temp1 = []
            temp2 = []
            for ij in range(len(attribute_dims)):
                pred = torch.softmax(temp_X[ij][:, 1:, :], dim=2).flatten(0, -2)
                true = Xs[ij][:, 1:].flatten()
                corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(device).reshape(-1,
                                                                                          temp_X[0].shape[1] - 1) * ( ~mask[:, 1:])
                corr_pred[corr_pred == 0] = 1
                cross_entropys = -torch.log(corr_pred)
                cross_entropy_max = cross_entropys.max(1).values.unsqueeze(1)  ##最大的损失
                temp1.append(cross_entropys)
                temp2.append(cross_entropy_max)

            temp1 = torch.cat(temp1, 1)
            temp2 = torch.cat(temp2, 1)

            trace_level_abnormal_score = end2end_decoder(torch.cat((torch.cat(s,1), temp1,temp2),1))

            trace_level_abnormal_scores.append(trace_level_abnormal_score.detach().cpu())

            s, enc_output = reconstruct_encoder(Xs)
            reconstruct_X = reconstruct_decoder(Xs, s, enc_output)

            for attr_index in range(len(attribute_dims)):
                reconstruct_X[attr_index] = reconstruct_X[attr_index]
                reconstruct_X[attr_index] = torch.softmax(reconstruct_X[attr_index],dim=2)

            mask[:, 0] = True  # 第一个事件是我们添加的起始事件，屏蔽掉

            for attr_index in range(len(attribute_dims)):
                truepos = Xs[attr_index].flatten()
                p = reconstruct_X[attr_index].reshape((truepos.shape[0], -1)).gather(1, truepos.view(-1, 1)).squeeze()
                p_distribution = reconstruct_X[attr_index].reshape((truepos.shape[0], -1))

                p_distribution = p_distribution + 1e-8  #避免出现概率为0

                attr_level_abnormal_scores[pos: pos + Xs[attr_index].shape[0], :,attr_index] = \
                    (torch.sigmoid((torch.sum(torch.log(p_distribution) * p_distribution, 1) - torch.log(p)).reshape(
                        Xs[attr_index].shape)) * (~mask)).detach().cpu()
            pos += Xs[attr_index].shape[0]

        event_level_abnormal_scores = attr_level_abnormal_scores.max((2))

        trace_level_abnormal_scores = torch.cat(trace_level_abnormal_scores, dim=0).flatten()

        return  trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores
