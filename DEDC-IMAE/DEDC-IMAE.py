import argparse
import datetime
import logging
import pickle
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn.parameter import Parameter
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from evaluation import eva, eva3
import warnings

warnings.filterwarnings("ignore")


class IMAE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):

        super(IMAE, self).__init__()
        self.topic_num = args.topic_num

        text_to_hideen = []
        for i in range(self.topic_num):
            encoder_layer = nn.Sequential(
                *block(n_input, n_enc_1),
                *block(n_enc_1, n_enc_2),
                *block(n_enc_2, n_enc_3), )
            text_to_hideen.append(encoder_layer)

        self.text_to_hidden = nn.ModuleList(text_to_hideen)  #

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
        # nn.RReLU(num_parameters=1, init=0.25)
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


import torch
import torch.nn.functional as F


def compute_match_loss(prediction, target, device):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mse_losses = []
    for i in range(len(prediction)):
        mse = F.mse_loss(prediction[i].to(device), torch.tensor(target[i]).to(device))
        mse_losses.append(mse.item())
    average_mse = sum(mse_losses) / len(mse_losses)
    # mse_losses = []
    # for i in range(len(prediction)):
    #     target = torch.tensor(target[i]).expand(prediction[i].size()[0], -1)
    #     mse = F.mse_loss(prediction[i], target.to(device))
    #     mse_losses.append(mse.item())
    # average_mse = sum(mse_losses) / len(mse_losses)

    # mse = F.mse_loss(prediction, torch.tensor(target).to(device))

    return average_mse


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_IMAE(model, time, dataset, y, inherit, kmeans_mu_history):
    print("The datasets on {}-th slice is pretraining".format(time))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)  # 2e-4
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    # lr = 0.0002, weight_decay = 0.0001
    # lr = 0.0002, weight_decay = 0.001
    best_val_accuracy = 0.0
    acclist = []
    topic = ''
    best_model = None
    end_epoch = 300
    for epoch in range(1, end_epoch + 1):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):

            x = x.to(device)
            x_bar, _, Muti_C = model(x)
            re_loss = F.mse_loss(x_bar, x)

            if inherit != 0:
                ma_loss = compute_match_loss(Muti_C, kmeans_mu_history, device)
                loss = 1 * re_loss + 0.1 * ma_loss
                # loss = 1 * re_loss
            else:
                loss = re_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).to(device).float()
            x_bar, Mixed_z, Muti_C = model(x)
            loss = F.mse_loss(x_bar, x)
            # print('{} loss: {}'.format(epoch, loss))
            if len(kmeans_mu_history) == 0:
                kmeans = KMeans(n_clusters=n_clusters, n_init=30, random_state=42).fit(
                    Mixed_z.data.cpu().numpy())
            else:
                kmeans = KMeans(n_clusters=n_clusters, n_init=30, random_state=42, init=kmeans_mu_history).fit(
                    Mixed_z.data.cpu().numpy())
            center = kmeans.cluster_centers_
            acc = eva3(y, kmeans.labels_, epoch)

            acclist.append(acc)
            if acc > best_val_accuracy:
                best_val_accuracy = acc
                topic = center
                best_model = deepcopy(model.state_dict())
            if epoch == end_epoch:
                with open(args.inherit_topic_path, 'wb') as f:
                    pickle.dump(topic, f)
                torch.save(best_model, args.pretrain_path)
                print('save: done')

            # eva(y, kmeans.labels_, epoch)
    # Epochs=list(range(1,101))
    # plt.plot(Epochs,acclist,marker='o', linestyle='-')
    # plt.xlabel('Epoch')
    # plt.ylabel('ACC')
    # plt.title('pretraining ACC over Time{}'.format(time))
    # plt.show()


def append_to_excel(filename, data):
    try:
        df = pd.read_excel(filename, engine='openpyxl')
    except FileNotFoundError:
        df = pd.DataFrame()

    df = df.append(data, ignore_index=True)
    df.to_excel(filename, index=False, engine='openpyxl')


def train_dedc(time, dataset, y, inherit, kmeans_mu_history, device):
    model = DEDC(500, 500, 2000, 2000, 500, 500, n_input=args.n_input, n_z=args.n_z, n_clusters=args.n_clusters, v=1.0,
                 pretrain_path=args.pretrain_path).to(device)

    model.pretrain(time, dataset, y, inherit, kmeans_mu_history, device)

    # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.5)
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    with torch.no_grad():
        # Xbar
        # Mixed_Z
        Xbar, Mixed_Z, Muti_C = model.imae(data)
    if len(kmeans_mu_history) == 0:
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=30, random_state=42)
    else:
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=30, random_state=42, init=kmeans_mu_history)
    y_pred = kmeans.fit_predict(Mixed_Z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')
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
    print(f'loss={args.kl_weight}*kl_loss + {args.re_weight}*re_loss + {args.ma_weight}*ma_loss')
    for epoch in range(200):
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
            # acclist.append(accp)

            # if epoch > 0 and delta_label < args.tol:
            #     print('delta_label {:.4f}'.format(delta_label), '< tol',
            #           args.tol)
            #     print('Reached tolerance threshold. Stopping training.')
            #     break
        #
        # for batch_idx, (x, idx) in enumerate(train_loader):
        #     x = x.to(device)
        #     idx = idx.to(device)
        #     Xbar, tmp_q, Mixed_Z, Muti_C = model(x)
        #     kl_loss = F.kl_div(tmp_q.log(), p, reduction='batchmean')
        #     re_loss = F.mse_loss(Xbar, x)
        #     if inherit != 0:
        #         inherit_loss = compute_match_loss(model.cluster_layer.data, kmeans_mu_history, device)
        #         ma_loss = compute_match_loss(Muti_C, kmeans_mu_history, device)
        #         loss = 0.02 * kl_loss + re_loss + 0.01 * inherit_loss + 0.01 * ma_loss
        #     else:
        #         loss = re_loss + 0.02 * kl_loss
        #
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        Xbar, tmp_q, Mixed_Z, Muti_C = model(data)

        kl_loss = F.kl_div(tmp_q.log(), p, reduction='batchmean')
        # ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(Xbar, data)
        if inherit != 0:
            ma_loss = compute_match_loss(model.cluster_layer.data, kmeans_mu_history, device)

            loss = args.kl_weight * kl_loss + args.re_weight * re_loss + args.ma_weight * ma_loss

        else:
            # loss = 0.01 * kl_loss + 1*re_loss
            loss = args.kl_weight * kl_loss + args.re_weight * re_loss

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        # scheduler.step()

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

        self.pretrain_path = pretrain_path
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v

    def forward(self, x):
        Xbar, Mixed_Z, Muti_C = self.imae(x)
        q = 1.0 / (1.0 + torch.sum(torch.pow(Mixed_Z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return Xbar, q, Mixed_Z, Muti_C

    def pretrain(self, time, dataset, y, inherit, kmeans_mu_history, device):
        if PRETRAIN:
            # if not os.path.exists(self.pretrain_path) and PRETRAIN:
            self.imae.load_state_dict(torch.load(args.pretrain_path, map_location=device))
            pretrain_IMAE(self.imae, time, dataset, y, inherit, kmeans_mu_history)
        # load pretrain weights
        # self.imae.load_state_dict(torch.load(self.pretrain_path),map_location=device)
        self.imae.load_state_dict(torch.load(args.pretrain_path, map_location=device))
        print('load pretrained imae from', args.pretrain_path)


class LoggerWriter:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.rstrip():
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass


def use_log(flag, log_time):
    if flag:
        logging.basicConfig(filename=f'lab/log/{dataset_name}_{log_time}', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)


if __name__ == '__main__':
    time_now = datetime.datetime.now().__format__('%b_%d_%Y_%H_%M')
    second = datetime.datetime.now().timestamp()
    dataset_name = 'NCA'
    PRETRAIN = True
    use_log(flag=True, log_time=time_now)
    # while True:
    for time in range(3, 4):  # time:1-4

        # NCA
        datapath = 'NCA/paperseriestime{}.txt'.format(time)
        label_path = 'NCA/paperseriestime{}_label.txt'.format(time)

        x = np.loadtxt(datapath, dtype=float)
        y = np.loadtxt(label_path, dtype=int)
        dataset = LoadDataset(x)
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default=datapath)
        # parser.add_argument('--k', type=int, default=3)
        parser.add_argument('--lr', type=float, default=0.0002)
        parser.add_argument('--n_clusters', default=3, type=int)
        parser.add_argument('--n_z', default=10, type=int)
        parser.add_argument('--n_input', default=10000, type=int)
        parser.add_argument('--tol', default=0.002, type=float)
        parser.add_argument('--update_interval', default=10, type=int)
        parser.add_argument('--batch_size', default=128)
        parser.add_argument('--kl_weight', default=0.1)
        parser.add_argument('--re_weight', default=1)
        parser.add_argument('--ma_weight', default=0.1)
        parser.add_argument('--topic_num', default=3)
        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))
        device = torch.device("cuda:0" if args.cuda else "cpu")
        args.pretrain_path = f'lab/pretrain/{dataset_name}/pretrain_time{time}.pkl'
        args.inherit_topic_path = f'lab/inherit/{dataset_name}/inherit_topic{time}.pkl'

        print(args)
        if time == 1:
            inherit = 0  # k centers of k-means algorithm
            kmeans_mu_history = []
        else:
            inherit = 1
            with open(f'lab/inherit/{dataset_name}/inherit_topic{time - 1}.pkl', 'rb') as f:
                kmeans_mu_history = pickle.load(f)
        train_dedc(time, dataset, y, inherit, kmeans_mu_history, device)
