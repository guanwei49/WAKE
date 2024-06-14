import argparse
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_json import Dataset_json
from model.detect import detect
from model.train import train_phase1, train_phase2, train_phase3

from dataset import Dataset
import torch.utils.data as Data

from utils.eval import cal_best_PRF
from utils.fs import EVENTLOG_DIR


def main(dataset, beta=0.3, batch_size=64, n_epochs_1=10, n_epochs_2=10, n_epochs_3=4, p_lambda=10, lr=0.0002, b1=0.5, b2=0.999, seed=None, enc_hidden_dim = 64, encoder_num_layers = 4, decoder_num_layers=2, dec_hidden_dim = 64):
    '''
    :param dataset:  Dataset(class)
    :param beta: Control the ratio of labeled anomalies to unlabeled samples
    :param batch_size:
    :param n_epochs_1: epoch of pre-training stage
    :param n_epochs_2: epoch of end-to-end  optimization stage
    :param n_epochs_3: epoch of fine-tuning stage
    :param p_lambda: a hyper-parameter to balance the contributions of two parts to the joint loss function
    :param lr: learning rate
    :param b1: adam: decay of first order momentum of gradient
    :param b2: adam: decay of first order momentum of gradient
    :param seed: value of Pytorch random seed
    :param enc_hidden_dim: hidden dimensions of BGRU layers in the encoder of feature encoder
    :param encoder_num_layers: number of BGRU layers in the encoder of feature encoder
    :param decoder_num_layers: number of GRU layers in the decoder of feature encoder
    :param dec_hidden_dim: hidden dimensions of GRU layers in the decoder of feature encoder
    :return:
    '''

    Xs_clean = []
    for i, dim in enumerate(dataset.attribute_dims):
        Xs_clean.append( torch.LongTensor(np.delete(dataset.features[i], dataset.labeled_indices, 0)))
    clean_mask=torch.BoolTensor(np.delete(dataset.mask, dataset.labeled_indices, 0))
    clean_dataset = Data.TensorDataset(*Xs_clean, clean_mask)
    clean_dataloader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                                  drop_last=True)
    encoder, decoder=train_phase1(clean_dataloader, dataset.attribute_dims,n_epochs_1, lr, b1, b2, seed,
                                enc_hidden_dim,encoder_num_layers, decoder_num_layers,dec_hidden_dim)

    anomalies_num=int(dataset.weak_labels.sum())
    repeat_times=int((len(dataset.weak_labels) * beta) / ((1 - beta) * anomalies_num))
    train_Xs = []
    for i, dim in enumerate(dataset.attribute_dims):
        train_Xs.append( torch.LongTensor(dataset.features[i]))
    train_labels = torch.LongTensor(dataset.weak_labels)
    for i in dataset.labeled_indices:
        for j, dim in enumerate(dataset.attribute_dims):
            train_Xs[j] = torch.cat((train_Xs[j], train_Xs[j][i].repeat((repeat_times, 1))))
    train_mask =torch.BoolTensor(dataset.mask)
    for i in dataset.labeled_indices:
        train_mask = torch.cat((train_mask, train_mask[i].repeat((repeat_times, 1))))
    train_labels=torch.cat((train_labels,torch.ones(len(dataset.labeled_indices)*repeat_times)))
    train_dataset = Data.TensorDataset(*train_Xs,train_mask,train_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True, drop_last=True)

    end2end_decoder = train_phase2(train_dataloader, dataset.attribute_dims, dataset.max_len, encoder, decoder, p_lambda, n_epochs_2, lr, b1, b2, seed, dec_hidden_dim)

    Xs = []
    for i, dim in enumerate(dataset.attribute_dims):
        Xs.append(torch.LongTensor(dataset.features[i]))
    mask = torch.BoolTensor(dataset.mask)
    labels = torch.LongTensor(dataset.weak_labels)

    ori_dataset = Data.TensorDataset(*Xs, mask, labels)
    ori_dataloader = DataLoader(ori_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                                  drop_last=True)
    reconstruct_encoder, reconstruct_decoder = train_phase3(ori_dataloader,dataset.attribute_dims,encoder,decoder, n_epochs_3 ,lr ,b1 ,b2 ,seed)

    detect_dataset = Data.TensorDataset(*Xs, mask)

    detect_dataloader = DataLoader(detect_dataset, batch_size=batch_size,
                            shuffle=False,num_workers=0,pin_memory=True)

    attr_Shape=(dataset.num_cases,dataset.max_len,dataset.num_attributes)
    trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores = detect(encoder,decoder,end2end_decoder,reconstruct_encoder,reconstruct_decoder, detect_dataloader, dataset.attribute_dims,attr_Shape=attr_Shape)

    return trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--mode', type=str, default='test', help='specify the mode')

    args = parser.parse_args()

    if args.mode != 'eval':
        attr_keys = ['concept:name', 'org:resource', 'org:role']

        ROOT_DIR = Path(__file__).parent
        logPath = os.path.join(ROOT_DIR, 'BPIC20_PrepaidTravelCost.xes')
        labelPath = os.path.join(ROOT_DIR, 'labels.npy')
        dataset = Dataset(logPath, labelPath, attr_keys)
        trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores = main(dataset,
                                                                                                    beta=0.3,
                                                                                                    batch_size=64,
                                                                                                    n_epochs_1=14,
                                                                                                    n_epochs_2=10,
                                                                                                    n_epochs_3=4,
                                                                                                    lr=0.0002,
                                                                                                    p_lambda=10,
                                                                                                    encoder_num_layers=4,
                                                                                                    decoder_num_layers=2,
                                                                                                    enc_hidden_dim=64,
                                                                                                    dec_hidden_dim=64)
    else:  # evaluate the method using all the dataset in EVENTLOG_DIR
        print(EVENTLOG_DIR)
        if os.path.isdir(os.path.join(EVENTLOG_DIR, 'cache')) == False:
            os.mkdir(os.path.join(EVENTLOG_DIR, 'cache'))
        dataset_names = os.listdir(EVENTLOG_DIR)
        dataset_names.sort()
        dataset_names.remove('cache')
        print(dataset_names)

        resPath = f'eva_result.csv'
        for dataset_name in dataset_names:
            try:
                print(dataset_name)
                start_time = time.time()
                # dataset = Dataset(dataset_name,label_percent=label_percent)
                dataset = Dataset_json(dataset_name,
                                       label_percent=0.1)  # The 10% anomalies are treated as labelled anomalies.
                hidden_dim = 64
                n_epochs_2 = 10
                if 'Billing' in dataset_name or 'Declaration' in dataset_name:
                    hidden_dim = 16
                if 'Receipt' in dataset_name:
                    n_epochs_2 = 20
                trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores = main(dataset,
                                                                                                            beta=0.3,
                                                                                                            batch_size=64,
                                                                                                            n_epochs_1=14,
                                                                                                            n_epochs_2=n_epochs_2,
                                                                                                            n_epochs_3=4,
                                                                                                            lr=0.0002,
                                                                                                            p_lambda=10,
                                                                                                            encoder_num_layers=4,
                                                                                                            decoder_num_layers=2,
                                                                                                            enc_hidden_dim=hidden_dim,
                                                                                                            dec_hidden_dim=hidden_dim)

                end_time = time.time()

                run_time = end_time - start_time
                print(run_time)

                ##Anomaly Detection
                trace_truth = dataset.case_target
                trace_detected = trace_level_abnormal_scores
                trace_p, trace_r, trace_f1, trace_aupr = cal_best_PRF(trace_truth, trace_detected)
                print("Anomaly Detection")
                print(trace_p, trace_r, trace_f1, trace_aupr)  ##precision,recall,F1-score,AP

                ## Interpretation of the Cause of Anomalies
                attr_truth = dataset.binary_targets[trace_truth.astype('bool')][
                    ~dataset.mask[trace_truth.astype('bool')]]
                attr_detected = np.array(attr_level_abnormal_scores)[trace_truth.astype('bool')][
                    ~dataset.mask[trace_truth.astype('bool')]]
                attr_p, attr_r, attr_f1, attr_aupr = cal_best_PRF(attr_truth.flatten(),
                                                                  attr_detected.flatten())
                print("Interpretation of the Cause of Anomalies")
                print(attr_p, attr_r, attr_f1, attr_aupr)  ##precision,recall,F1-score,AP

                datanew = pd.DataFrame(
                    [{'index': dataset_name, 'AD_precision': trace_p, "AD_recall": trace_r, 'AD_f1': trace_f1,
                      'AD_aupr': trace_aupr,
                      'IA_precision': attr_p, "IA_recall": attr_r, 'IA_f1': attr_f1, 'IA_aupr': attr_aupr,
                      'times': run_time,
                      }])
                if os.path.exists(resPath):
                    data = pd.read_csv(resPath)
                    data = pd.concat([data, datanew], ignore_index=True)
                else:
                    data = datanew
                data.to_csv(resPath, index=False)
            except:
                traceback.print_exc()
                datanew = pd.DataFrame([{'index': dataset_name}])
                if os.path.exists(resPath):
                    data = pd.read_csv(resPath)
                    data = pd.concat([data, datanew], ignore_index=True)
                else:
                    data = datanew
                data.to_csv(resPath, index=False)


