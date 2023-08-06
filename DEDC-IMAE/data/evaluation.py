import numpy as np
from munkres import Munkres, print_matrix
import math
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear, linear_sum_assignment
from sklearn import metrics


def cluster_nmi1(y_true,y_pred):
    #样本点数
    toty_truel = len(y_true)
    y_true_ids = set(y_true)
    y_pred_ids = set(y_pred)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idy_true in y_true_ids:
        for idy_pred in y_pred_ids:
            idy_trueOccur = np.where(y_true==idy_true)
            idy_predOccur = np.where(y_pred==idy_pred)
            idy_truey_predOccur = np.intersect1d(idy_trueOccur,idy_predOccur)
            px = 1.0*len(idy_trueOccur[0])/toty_truel
            py = 1.0*len(idy_predOccur[0])/toty_truel
            pxy = 1.0*len(idy_truey_predOccur)/toty_truel
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idy_true in y_true_ids:
        idy_trueOccurCount = 1.0*len(np.where(y_true==idy_true)[0])
        Hx = Hx - (idy_trueOccurCount/toty_truel)*math.log(idy_trueOccurCount/toty_truel+eps,2)
    Hy = 0
    for idy_pred in y_pred_ids:
        idy_predOccurCount = 1.0*len(np.where(y_pred==idy_pred)[0])
        Hy = Hy - (idy_predOccurCount/toty_truel)*math.log(idy_predOccurCount/toty_truel+eps,2)
    MIhy_truet = 2.0*MI/(Hx+Hy)
    return MIhy_truet


def cluster_acc1(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size,w
def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    # acc, f1 = cluster_acc(y_true, y_pred)
    acc = cluster_acc1(y_true, y_pred)[0]
    nmi = cluster_nmi1(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    # nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    # ari = ari_score(y_true, y_pred)
    # print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
    #         ', f1 {:.4f}'.format(f1))
    # print(acc,nmi)
    print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
