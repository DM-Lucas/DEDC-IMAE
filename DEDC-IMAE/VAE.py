'''
   how to realise the substitute of prior to posterior BCE add weigh,
   Training one time slice at a time slice
   input:path,hyper-parameters
   output:four k-means centers and the model parameters of trained per time silce perserve to pk file .
'''

from __future__ import print_function, division
import pickle

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from evaluation import   eva3
import warnings
from torch.utils.data import Dataset

torch.cuda.set_device(0)

warnings.filterwarnings("ignore")

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
        )
        self.z_mean = nn.Linear(2000, latent_dim)
        self.z_log_var = nn.Linear(2000, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000,500),
            nn.ReLU(),
            nn.Linear(500, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        encoded = self.encoder(x)
        z_mean = self.z_mean(encoded)
        z_log_var = self.z_log_var(encoded)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon
        return z

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        decoded = self.decode(z)
        return decoded, z_mean, z_log_var,z

# -------------训练VAE模型
def train_vae(vae, train_loader, optimizer, loss_fn, device):
    vae.train()
    train_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, z_mean, z_log_var,z = vae(data.view(data.size(0), -1))
        loss = loss_fn(recon_batch, data, z_mean, z_log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss /= len(train_loader.dataset)
    return train_loss

# KL散度和重构误差的损失函数
def loss_fn(recon_x, x, z_mean, z_log_var):
    recon_loss = nn.BCELoss(reduction='sum')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return recon_loss + kl_loss

class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设置参数
    for t in range(4):  # time:1-4
        time = t + 1
        # -------
        # replace the dataset filename.
        datapath = 'NCA_noise/papernoiseseriestime{}.txt'.format(time)
        label_path = 'NCA_noise/papernoiseseriestime{}_label.txt'.format(time)
        x = np.loadtxt(datapath, dtype=float)
        y = np.loadtxt(label_path, dtype=int)
        dataset = LoadDataset(x)
        data = torch.Tensor(dataset.x).to(device)
        y = torch.Tensor(y).to(device)
        input_dim = 10000
        latent_dim = 3
        batch_size = 256
        epochs = 300
        lr = 0.0002

        dataloader = DataLoader(TensorDataset(data, y), batch_size=batch_size, shuffle=True)
        # ----train and save the kmeans centers----#

        # 初始化VAE模型和优化器

        vae = VAE(input_dim, latent_dim).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=lr)

        # 训练VAE模型
        for epoch in range(1, epochs + 1):
            train_loss = train_vae(vae, dataloader, optimizer, loss_fn, device)
            print('Epoch {}: Train Loss: {:.4f}'.format(epoch, train_loss))

        # 提取潜在表示
        vae.eval()
        latent_vectors = []
        Y = []
        with torch.no_grad():
            for data, y in dataloader:
                data = data.to(device)
                _, z_mean, _,z = vae(data.view(data.size(0), -1))
                latent_vectors.append(z_mean)
                Y.append(y)
        latent_vectors = torch.cat(latent_vectors).detach().cpu().numpy()
        Y = torch.cat(Y, 0).detach().cpu().numpy()

        with open('vaenewnoiseZ_mutime{}.pk'.format(time+1), 'wb') as f:

            pickle.dump(latent_vectors, f)

        # 使用K-means进行聚类
        kmeans = KMeans(n_clusters=3, n_init=30, random_state=42).fit(latent_vectors)
        center = kmeans.cluster_centers_

        # y is ndarry kmeans.labels_ is list?
        acc = eva3(Y, kmeans.labels_, epoch)

        # ddtm.predict(time+1,path)