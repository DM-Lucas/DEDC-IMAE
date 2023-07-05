import pickle

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva

torch.cuda.set_device(0)

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

def block(in_c,out_c):
    layers=[
        nn.Linear(in_c,out_c),
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

def match_loss(Muti_C,last_tpoic):

    match_loss = F.mse_loss(Muti_C, last_tpoic)

    return match_loss


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def compute_match_loss(target, prediction):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mse_losses = []
    for t, p in zip(target, prediction):
        t = t.to(device)
        p = p.to(device)
        mse_loss = F.mse_loss(p.unsqueeze(0).expand(t.size(0), -1), t, reduction='mean')
        mse_losses.append(mse_loss)
    loss = torch.mean(torch.stack(mse_losses))
    return loss




def pretrain_IMVE(model, dataset, y, inherit, kmeans_mu_history):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=2e-3)


    for epoch in range(1):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()
            x_bar, _,Muti_C = model(x)
            re_loss = F.mse_loss(x_bar, x)

            if inherit != 0:
                ma_loss = compute_match_loss(Muti_C, kmeans_mu_history)
                loss = re_loss + 0.01 * ma_loss
            else:
                loss = re_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, Mixed_z,Muti_C = model(x)
            loss = F.mse_loss(x_bar, x)


            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=3, n_init=30,random_state=42).fit(Mixed_z.data.cpu().numpy())
            eva(y, kmeans.labels_, epoch)

        torch.save(model.state_dict(), 'IMAEpaperseriestime{}.pkl'.format(time))

if __name__ == '__main__':
    # -------
    time = 2 #time epoch
    # -------
    imae = IMAE(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=10000,
        n_z=10,).cuda()

    x = np.loadtxt('paperseriestime{}.txt'.format(time), dtype=float)
    y = np.loadtxt('paperseriestime{}_label.txt'.format(time), dtype=int)

    dataset = LoadDataset(x)



    kmeans_mu_history = ''

    if time == 1:
        inherit = 0  # k centers of k-means algorithm
    else:
        inherit = 1
        with open('../kmeans_mu_history{}.pkl'.format(time - 1), 'rb') as f:
            kmeans_mu_history = pickle.load(f)

    pretrain_IMVE(imae, dataset, y, inherit, kmeans_mu_history)
