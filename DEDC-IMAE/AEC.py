'''
The method of clustering implement Aaaptively Evolutionary clustering.
We chose the base algorithm on K-means and initial the center of clustering with history clustering result.
The theory can be discover with the link https://arxiv.org/pdf/1104.1990.pdf
'''

from __future__ import print_function, division

import numpy as np
import torch
from sklearn.cluster import KMeans
from evaluation import evaaec
from torch.utils.data import Dataset
#GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dataset


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

def base_kmeanscon(time):
    # datapath = 'NCA/paperseriestime{}.txt'.format(time)
    # label_path = 'NCA/paperseriestime{}_label.txt'.format(time)

    # datapath = 'NCA_noise/papernoiseseriestime{}.txt'.format(time)
    # label_path = 'NCA_noise/papernoiseseriestime{}_label.txt'.format(time)

    # datapath = 'A5A/newsnoiseseriestime{}.txt'.format(time)
    # label_path = 'A5A/newsnoiseseriestime{}_label.txt'.format(time)

    # datapath = 'A5A/newsseriestime{}.txt'.format(time)
    # label_path = 'A5A/newsseriestime{}_label.txt'.format(time)

    datapath = 'NCA10/paperseriestime{}.txt'.format(time)
    label_path = 'NCA10/paperseriestime{}_label.txt'.format(time)

    print(datapath)
    x = np.loadtxt(datapath, dtype=float)
    y = np.loadtxt(label_path, dtype=int)

    dataset = LoadDataset(x)

    print("第" + str(time) + "时间片聚类")

    data = dataset.x

    #k-means
    kmeans = KMeans(n_clusters=3, n_init=30,random_state=42).fit(data)
    center = kmeans.cluster_centers_
    evaaec(y, kmeans.labels_)
    return center

def base_kmeans(time,center):
    # datapath = 'NCA/paperseriestime{}.txt'.format(time)
    # label_path = 'NCA/paperseriestime{}_label.txt'.format(time)

    # datapath = 'NCA_noise/papernoiseseriestime{}.txt'.format(time)
    # label_path = 'NCA_noise/papernoiseseriestime{}_label.txt'.format(time)

    # datapath = 'A5A/newsnoiseseriestime{}.txt'.format(time)
    # label_path = 'A5A/newsnoiseseriestime{}_label.txt'.format(time)

    datapath = 'NCA10/paperseriestime{}.txt'.format(time)
    label_path = 'NCA10/paperseriestime{}_label.txt'.format(time)
    print(datapath)
    x = np.loadtxt(datapath, dtype=float)
    y = np.loadtxt(label_path, dtype=int)

    dataset = LoadDataset(x)

    print("第" + str(time) + "时间片聚类")

    data = dataset.x

    #k-means
    kmeans = KMeans(n_clusters=3, n_init=30,random_state=42,init=center).fit(data)
    center = kmeans.cluster_centers_
    evaaec(y, kmeans.labels_)
    return center

if __name__ == '__main__':

    times=[1,2,3,4,5,6,7,8,9,10]
    # # # -----AEC
    for time in times:
        if time == 1:
            center=0
            center = base_kmeanscon(time)
        else:
            center = base_kmeans(time,center)
    # contrast experiments
    # for time in times:
    #     center = base_kmeanscon(time)
