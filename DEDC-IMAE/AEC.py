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
import time

class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def base_kmeanscon(item):

    # datapath = 'NCA/paperseriestime{}.txt'.format(time)
    # label_path = 'NCA/paperseriestime{}_label.txt'.format(time)

    # datapath = 'NCA_noise/papernoiseseriestime{}.txt'.format(time)
    # label_path = 'NCA_noise/papernoiseseriestime{}_label.txt'.format(time)

    # datapath = 'A5A_noise/newsnoiseseriestime{}.txt'.format(time)
    # label_path = 'A5A_noise/newsnoiseseriestime{}_label.txt'.format(time)

    # datapath = 'News5/news5time{}.txt'.format(time)
    # label_path = 'News5/news5time{}_label.txt'.format(time)
    # datapath = 'BBC/BBCtime{}.txt'.format(time)
    # label_path = 'BBC/BBCtime{}_label.txt'.format(time)
    # datapath = 'NCA10/paperseriestime{}.txt'.format(item)
    # label_path = 'NCA10/paperseriestime{}_label.txt'.format(item)

    datapath = 'WOB5/WOB5_{}.txt'.format(item)
    label_path = 'WOB5/WOB5_{}_label.txt'.format(item)


    print(datapath)
    x = np.loadtxt(datapath, dtype=float)
    y = np.loadtxt(label_path, dtype=int)

    dataset = LoadDataset(x)

    print("第" + str(item) + "时间片聚类")

    data = dataset.x

    #k-means
    kmeans = KMeans(n_clusters=5, n_init=30,random_state=42).fit(data)
    center = kmeans.cluster_centers_
    evaaec(y, kmeans.labels_)
    return center

def base_kmeans(item,center):
    # datapath = 'NCA/paperseriestime{}.txt'.format(time)
    # label_path = 'NCA/paperseriestime{}_label.txt'.format(time)

    # datapath = 'NCA_noise/papernoiseseriestime{}.txt'.format(time)
    # label_path = 'NCA_noise/papernoiseseriestime{}_label.txt'.format(time)

    # datapath = 'A5A_noise/newsnoiseseriestime{}.txt'.format(time)
    # label_path = 'A5A_noise/newsnoiseseriestime{}_label.txt'.format(time)

    # datapath = 'News5/news5time{}.txt'.format(item)
    # label_path = 'News5/news5time{}_label.txt'.format(item)

    datapath = 'WOB5/WOB5_{}.txt'.format(item)
    label_path = 'WOB5/WOB5_{}_label.txt'.format(item)
    # datapath = 'NCA10/paperseriestime{}.txt'.format(time)
    # label_path = 'NCA10/paperseriestime{}_label.txt'.format(time)
    print(datapath)
    x = np.loadtxt(datapath, dtype=float)
    y = np.loadtxt(label_path, dtype=int)

    dataset = LoadDataset(x)

    print("第" + str(slice) + "时间片聚类")

    data = dataset.x

    #k-means
    kmeans = KMeans(n_clusters=5, n_init=30,random_state=42,init=center).fit(data)
    center = kmeans.cluster_centers_
    evaaec(y, kmeans.labels_)
    return center

if __name__ == '__main__':

    # start_time = time.time()
    items=[1,2,3,4]
    # time=4
    # -----AEC
    # for item in items:
    #     if item == 1:
    #         center=0
    #         center = base_kmeanscon(item)
    #     else:
    #         center = base_kmeans(item,center)

    # # # k-means
    for item in items:
        center = base_kmeanscon(item)

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"代码运行时间：{elapsed_time} 秒")