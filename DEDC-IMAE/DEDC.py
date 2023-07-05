from __future__ import print_function, division
import argparse
import pickle
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph

from evaluation import eva
from collections import Counter

torch.cuda.set_device(1)

def block(in_c,out_c):
    layers=[
        nn.Linear(in_c,out_c),
        nn.ReLU(True)
    ]
    return layers

class IMAE(nn.Module):

    def __init__(self,n_enc_1,n_enc_2,n_enc_3,n_dec_1,n_dec_2,n_dec_3,n_input,n_z):

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

        Muti_C=[]

        for k in range(self.topic_num):
            C_k = self.multi_C[k](self.text_to_hidden[k](X))

            Muti_C.append(C_k)


        Mixed_Z = self.attention(torch.cat(Muti_C, dim=-1))
        Xbar = self.decoder_layer(Mixed_Z)

        return Xbar,Mixed_Z,Muti_C

class DEDC(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,n_dec_1, n_dec_2,n_dec_3,
                 n_input, n_z, n_clusters, v=1):

        super(DEDC, self).__init__()

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

        self.imae.load_state_dict(torch.load('data/IMAEpaperseriestime1.pkl', map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x):
        # DNN Module

        # print('%'*100)
        Xbar,Mixed_Z,_= self.imae(x)
        q = 1.0 / (1.0 + torch.sum(torch.pow(Mixed_Z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return Xbar, q,Mixed_Z


def target_distribution(q):

    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_DEDC(dataset, inherit, kmeans_mu_history):

    model = DEDC(500,500,2000,2000,500,500, n_input=args.n_input, n_z=args.n_z, n_clusters=args.n_clusters, v=1.0).to(device)

    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    with torch.no_grad():

        Xbar,Mixed_Z,Muti_C = model.imae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20,random_state=42)
    y_pred = kmeans.fit_predict(Mixed_Z.data.cpu().numpy())

    eva(y, y_pred, 'pae')

    #-------initial historical centers-------#

    #initialize
    if inherit != 0:
        model.cluster_layer.data = torch.tensor(kmeans_mu_history).to(device)
    else:
        model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    maxacc=0
    maxcenter = np.zeros((3, args.n_z), np.float32)
    for epoch in range(1):
        if epoch % 1 == 0:
            # update_interval
            Xbar,tmp_q,Mixed_Z= model(data)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            # res2 = pred.data.cpu().numpy().argmax(1)  # Z
            res3 = p.data.cpu().numpy().argmax(1)  # P

            _ = eva(y, res1, str(epoch) + 'Q')
            acc= eva(y, res3, str(epoch) + 'P')

        Xbar, q, Mixed_Z, = model(data)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')

        re_loss = nn.MSELoss()(Xbar, data)

        # iter mu and prior mu
        if inherit != 0:
            ma_loss = F.mse_loss(model.cluster_layer.data,kmeans_mu_history)
            loss = 0.01*kl_loss + 0.1 * ma_loss+re_loss
        else:
            loss = 0.01*kl_loss+re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if acc > maxacc:
            maxacc = acc
            maxcenter = model.cluster_layer.data

    mu_c = maxcenter  # np.array

    with open('kmeans_mu_history{}.pkl'.format(time), 'wb') as f:
        pickle.dump(mu_c, f)
    return mu_c


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='paperseriestime1')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(args.name)

    if args.name == 'paperseriestime1':
        args.lr = 2e-4
        args.k = None
        args.n_clusters = 3
        args.n_input = 10000
        args.n_z = 10

    print(args)

    #-------
    time = 2
    #-------
    kmeans_mu_history=''
    if time == 1:
        inherit = 0  # k centers of k-means algorithm
    else:
        inherit = 1
        with open('kmeans_mu_history{}.pkl'.format(time-1), 'rb') as f:
            kmeans_mu_history = pickle.load(f)

    train_DEDC(dataset,inherit,kmeans_mu_history)