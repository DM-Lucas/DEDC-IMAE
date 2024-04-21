"""
pytorch implementation for DEDC-IMAE algorithm:

Usage:
    use `python DEDC-IMAE.py -h` for help.

Author:
    Hui Lu. 2024.1.13
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva, eva3
import time
import torch
import torch.nn.functional as F
start_time = time.time()
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

class IMAE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(IMAE, self).__init__()
        self.topic_num = 3
        text_to_hideen = []
        for i in range(self.topic_num):
            encoder_layer = nn.Sequential(
                *block(n_input, n_enc_1),
                *block(n_enc_1, n_enc_2),
                *block(n_enc_2, n_enc_3), )
            text_to_hideen.append(encoder_layer)
        self.text_to_hidden = nn.ModuleList(text_to_hideen)
        multi_C = []
        for i in range(self.topic_num):
            multi_C.append(nn.Linear(n_enc_3, n_z))
        self.multi_C = nn.ModuleList(multi_C)
        self.mixed_dim = self.topic_num * n_z
        self.attention = nn.Linear(self.mixed_dim, n_z)
        self.decoder_layer = nn.Sequential(
            *block(n_z, n_dec_1),
            *block(n_dec_1, n_dec_2),
            *block(n_dec_2, n_dec_3),
            nn.Linear(n_dec_3, n_input),
        )
    def forward(self, X):

        Muti_C = []

        for k in range(self.topic_num):
            C_k = self.multi_C[k](self.text_to_hidden[k](X))
            Muti_C.append(C_k)

        Mixed_Z = self.attention(torch.cat(Muti_C, dim=-1))
        Xbar = self.decoder_layer(Mixed_Z)

        return Xbar, Mixed_Z, Muti_C

def block(in_c, out_c):
    layers = [
        nn.Linear(in_c, out_c),
        nn.ReLU(True)
    ]
    return layers

class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def match_loss(Muti_C, last_tpoic):
    match_loss = F.mse_loss(Muti_C, last_tpoic)
    return match_loss

def compute_match_loss(prediction, target, device):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mse_losses = []
    for i in range(len(prediction)):
        mse = F.mse_loss(prediction[i].to(device), torch.tensor(target[i]).to(device))
        mse_losses.append(mse.item())
    average_mse = sum(mse_losses)

    return average_mse

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def pretrain_IMAE(model, time, dataset, y, inherit, kmeans_mu_history):
    print("The datasets on {}-th slice is pretraining".format(time))
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    # print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val_accuracy = 0.0
    acclist = []

    maxacc = 0
    maxcenter = np.zeros((3, 3), np.float32)
    for epoch in range(400):
        # adjust_learning_rate(optimizer, epoch)
        total_loss = 0.
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_bar, _, Muti_C = model(x)
            re_loss = F.mse_loss(x_bar, x)
            if inherit != 0:
                # ma_loss = compute_match_loss(Muti_C, kmeans_mu_history,device) #list3,ndarray(3,10)
                # loss = re_loss + args.alpha * ma_loss
                loss = re_loss
            else:
                loss = re_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            x = torch.Tensor(dataset.x).to(device).float()
            x_bar, Mixed_z, Muti_C = model(x)
            # loss = F.mse_loss(x_bar, x)
            # print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=3, n_init=30, random_state=42).fit(Mixed_z.data.cpu().numpy())
            center = kmeans.cluster_centers_
            acc = eva3(y, kmeans.labels_, epoch)

            if acc > maxacc:
                maxacc = acc
                maxaccZ = Mixed_z
                # maxcenter = center
                topic = center

            # eva(y, kmeans.labels_, epoch)
    # Epochs=list(range(1,301))
    # plt.plot(Epochs,acclist,marker='o', linestyle='-')
    # plt.xlabel('Epoch')
    # plt.ylabel('ACC')
    # plt.title('pretraining ACC over Time{}'.format(time))
    # plt.show()

    with open('NCA/MAE_time{}.pkl'.format(time), 'wb') as f:
        pickle.dump(maxaccZ, f)
    with open(args.inherit_topic_path, 'wb') as f:
        pickle.dump(topic, f)
    torch.save(model.state_dict(), args.pretrain_path)

def append_to_excel(filename, data):
    # 如果文件存在，则读取原有数据
    try:
        df = pd.read_excel(filename, engine='openpyxl')
    except FileNotFoundError:
        df = pd.DataFrame()

    # 追加新数据并保存
    df = df.append(data, ignore_index=True)
    df.to_excel(filename, index=False, engine='openpyxl')

def train_dedc(time, dataset, y, inherit, kmeans_mu_history, device):

    model = DEDC(500, 500, 2000, 2000, 500, 500, n_input=args.n_input, n_z=args.n_z, n_clusters=args.n_clusters, v=1.0,
                 pretrain_path=args.pretrain_path).to(device)

    model.pretrain(time, dataset, y, inherit, kmeans_mu_history, device)

    return
    # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)

    with torch.no_grad():
        Xbar, Mixed_Z, Muti_C = model.imae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=42)
    y_pred = kmeans.fit_predict(Mixed_Z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    nmi_k = nmi_score(y_pred, y)
    print("nmi score={:.4f}".format(nmi_k))

    y_pred_last = y_pred

    # #-------initial historical centers-------#
    # #initialize
    # #inherit != 0 is difficute for training.
    # if inherit != 0:
    #     model.cluster_layer.data = torch.tensor(kmeans_mu_history).to(device)
    # else:
    #     model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model.train()
    maxacc = 0
    maxcenter = np.zeros((3, args.n_z), np.float32)
    acclist = []
    for epoch in range(100):
        if epoch % args.update_interval == 0:
            # update_interval
            Xbar, tmp_q, Mixed_Z, Muti_C = model(data)
            predict = F.softmax(Mixed_Z, dim=1)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            y_pre1 = tmp_q.cpu().numpy().argmax(1)  # Q
            # res2 = predict.data.cpu().numpy().argmax(1)  # Z
            y_pre2 = p.data.cpu().numpy().argmax(1)  # P
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            accq = eva(y, y_pre1, str(epoch) + 'Q')
            # accz =eva(y, res2, str(epoch) + 'Z')
            accp = eva(y, y_pre2, str(epoch) + 'P')
            acclist.append(accp)


        Xbar, tmp_q, Mixed_Z, Muti_C = model(data)

        Lc_loss = F.kl_div(tmp_q.log(), p, reduction='batchmean')
        # ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(Xbar, data)
        if inherit != 0:
            # inh_loss = compute_match_loss(model.cluster_layer.data, kmeans_mu_history, device)  # list3,ndarray(3,10)
            # loss = 0.1 * Lc_loss + 1 * re_loss + 0.1 * inh_loss  # DEDC
            loss = re_loss  # imae
        else:
            # loss = 0.1 * Lc_loss + 1 * re_loss
            loss = re_loss  # imae

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Epochs=list(range(1,101))
    # plt.plot(Epochs,acclist,marker='o', linestyle='-')
    # plt.xlabel('Epoch')
    # plt.ylabel('ACC')
    # plt.title('Training DEDC over Time{}'.format(time))
    # plt.show()
    #     if acc > maxacc:
    #         maxacc = acc
    #         maxcenter = model.cluster_layer.data
    #
    # mu_c = maxcenter  # np.array
    #
    # with open('kmeans_mu_history{}.pkl'.format(time), 'wb') as f:
    #     pickle.dump(mu_c, f)
    # return mu_c

class DEDC(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, pretrain_path=''):
        super(DEDC, self).__init__()
        self.pretrain_path = pretrain_path
        # autoencoder for intra information
        self.imae = IMAE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x):
        # DNN Module
        # print('%'*100)
        Xbar, Mixed_Z, Muti_C = self.imae(x)
        q = 1.0 / (1.0 + torch.sum(torch.pow(Mixed_Z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return Xbar, q, Mixed_Z, Muti_C

    def pretrain(self, time, dataset, y, inherit, kmeans_mu_history, device):
        if not os.path.exists(self.pretrain_path):
            pretrain_IMAE(self.imae, time, dataset, y, inherit, kmeans_mu_history)
        # load pretrain weights
        # self.imae.load_state_dict(torch.load(self.pretrain_path),map_location=device)
        self.imae.load_state_dict(torch.load(self.pretrain_path, map_location=device))
        print('load pretrained imae from', args.pretrain_path)


if __name__ == '__main__':

    for t in range(4):  # time:1-4
        item = t + 1
        # -------
        # replace the dataset filename.

        datapath = 'NCA/paperseriestime{}.txt'.format(item)
        label_path = 'NCA/paperseriestime{}_label.txt'.format(item)

        # datapath = 'NCA10/paperseriestime{}.txt'.format(time)
        # label_path = 'NCA10/paperseriestime{}_label.txt'.format(time)
        # datapath = 'NCA_noise/papernoiseseriestime{}.txt'.format(time)
        # label_path = 'NCA_noise/papernoiseseriestime{}_label.txt'.format(time)
        # datapath = 'A5A_noise/newsnoiseseriestime{}.txt'.format(time)
        # label_path = 'A5A_noise/newsnoiseseriestime{}_label.txt'.format(time)
        #
        # datapath = 'A5A/newsseriestime{}.txt'.format(time)
        # label_path = 'A5A/newsseriestime{}_label.txt'.format(time)

        x = np.loadtxt(datapath, dtype=float)
        y = np.loadtxt(label_path, dtype=int)
        dataset = LoadDataset(x)
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default=datapath)
        parser.add_argument('--k', type=int, default=3)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--n_clusters', default=3, type=int)
        parser.add_argument('--n_z', default=10, type=int)
        parser.add_argument('--n_input', default=10000, type=int)
        parser.add_argument('--tol', default=0.001, type=float)
        parser.add_argument('--update_interval', default=10, type=int)
        parser.add_argument(
            '--alpha',
            default=0.01,
            type=float,
            help='coefficient of matching loss')
        parser.add_argument(
            '--beta',
            default=0.1,
            type=float,
            help='coefficient of inherited loss')
        parser.add_argument(
            '--gamma',
            default=0.1,
            type=float,
            help='coefficient of clustering loss')

        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))
        device = torch.device("cuda:0" if args.cuda else "cpu")

        # args.pretrain_path = ''.format(time)

        # args.pretrain_path = 'NCA10/pretrain_time{}.pkl'.format(time)
        # args.inherit_topic_path='NCA10/inherit_topic{}.pkl'.format(time)

        args.pretrain_path = 'NCA/pretrain_time{}.pkl'.format(item)
        args.inherit_topic_path='NCA/inherit_topic{}.pkl'.format(item)

        # args.pretrain_path = 'NCA_noise/pretrain_time{}.pkl'.format(time)
        # args.inherit_topic_path='NCA_noise/inherit_topic{}.pkl'.format(time)

        # args.pretrain_path = 'A5A_noise/pretrain_time{}.pkl'.format(time)
        # args.inherit_topic_path='A5A_noise/inherit_topic{}.pkl'.format(time)

        # args.pretrain_path = 'A5A/pretrain_time{}.pkl'.format(time)
        # args.inherit_topic_path = 'A5A/inherit_topic{}.pkl'.format(time)

        # print(args)
        # # -------
        # time = 1
        # # -------
        # kmeans_mu_history = ''
        if item == 1:
            inherit = 0  # k centers of k-means algorithm
            kmeans_mu_history = 0
        else:
            inherit = 1
            with open('NCA/inherit_topic{}.pkl'.format(item - 1), 'rb') as f:
                kmeans_mu_history = pickle.load(f)
        train_dedc(item, dataset, y, inherit, kmeans_mu_history, device)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"代码运行时间：{elapsed_time} 秒")
