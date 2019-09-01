
# coding: utf-8
# probability prediction is saved in 'test_pb.txt'
# class prediction is saved in 'test_one_hot.txt'
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_similarity_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# # Util

# read feature
def read_feature():
    num_feature = 1097
    MF1, MF2, MF3 = np.zeros((1,num_feature)), np.zeros((1,num_feature)), np.zeros((1,num_feature))
    two_imgs_list = []
    six_imgs_list = []
    size = 66
    for i in range(size):
        fname = '%d.txt'%(i+1)
        tmp = np.loadtxt('testFeature/'+fname).astype(np.float)
        MF1 = np.concatenate((MF1, tmp[0].reshape(1,-1)), axis=0)
        MF2 = np.concatenate((MF2, tmp[1].reshape(1,-1)), axis=0)
        if tmp.shape[0] == 2:
            MF3 = np.concatenate((MF3, tmp[1].reshape(1,-1)), axis=0)
            two_imgs_list.append(i)
        else:
            MF3 = np.concatenate((MF3, tmp[2].reshape(1,-1)), axis=0)
        if tmp.shape[0] == 6:
            six_imgs_list.append(i)
    MF1 = MF1[1:, :]
    MF2 = MF2[1:, :]
    MF3 = MF3[1:, :]

    for i in six_imgs_list:
        if i%2 == 1:
            fname = label['ENSGID'].values[i].encode('ascii','ignore').strip(string.punctuation)+'.txt'
            tmp = np.loadtxt('testFeature/'+fname).astype(np.float)
            MF1[i] = tmp[3]
            MF2[i] = tmp[4]
            MF3[i] = tmp[5]

#     MF1 = preprocessing.scale(MF1)
#     MF2 = preprocessing.scale(MF2)
#     MF3 = preprocessing.scale(MF3)

    return MF1, MF2, MF3, two_imgs_list, six_imgs_list


MF1, MF2, MF3, two_imgs_list, six_imgs_list = read_feature()

# load
F1 = np.load('./npy/MF1.npy')
F2 = np.load('./npy/MF2.npy')
F3 = np.load('./npy/MF3.npy')
label = np.load('./npy/label.npy')[:,1:]

all_feature = preprocessing.scale(np.concatenate((MF1, MF2, MF3, F1, F2,F3), axis=0))

# test
MF1 = all_feature[0:66]
MF2 = all_feature[66:132]
MF3 = all_feature[132:198]
# train
F1 = all_feature[198:698]
F2 = all_feature[698:1198]
F3 = all_feature[1198:1698]



# PCA
pca = PCA(random_state=0)
pca.fit(all_feature)
num_pc = 100

# train
F1_pca = pca.transform(F1)[:, :num_pc]
F2_pca = pca.transform(F2)[:, :num_pc]
F3_pca = pca.transform(F3)[:, :num_pc]
# test
MF1_pca = pca.transform(MF1)[:, :num_pc]
MF2_pca = pca.transform(MF2)[:, :num_pc]
MF3_pca = pca.transform(MF3)[:, :num_pc]


# train

num_chain = 20
pred_train_pb, pred_test_pb = [], []
test_pca = [MF1_pca, MF2_pca, MF3_pca]
train_pca = [F1_pca, F2_pca, F3_pca]
for i in range(3):
    X_train, Y_train = train_pca[i], label
    X_test = test_pca[i]
    
    chains = [ClassifierChain(LogisticRegression(C=0.01, penalty='l2'), 
                          order='random', random_state=i) for i in range(num_chain)]
    
    for i, chain in enumerate(chains):
        chain.fit(X_train, Y_train)

    # predict
    pred_train_chains_pb = np.array([chain.predict_proba(X_train) for chain in chains])
    pred_train_ensemble_pb = pred_train_chains_pb.mean(axis=0)
#     pred_train_ens = np.argmax(pred_train_ensemble_pb, axis=1)

    pred_test_chains_pb = np.array([chain.predict_proba(X_test) for chain in chains])
    pred_test_ensemble_pb = pred_test_chains_pb.mean(axis=0)
#     pred_test_ens = np.argmax(pred_test_ensemble_pb, axis=1)

    pred_train_pb.append(pred_train_ensemble_pb)
    pred_test_pb.append(pred_test_ensemble_pb)

pred_train_pb = np.array(pred_train_pb)
pred_test_pb = np.array(pred_test_pb)


# submit probability
res_train_pb = np.max(pred_train_pb, axis=0)
res_test_pb = np.max(pred_test_pb, axis=0)
res_test_pb[29] = np.max(np.concatenate((pred_test_pb[0][29].reshape(1,-1), pred_test_pb[1][29].reshape(1,-1)), axis=0),axis=0)



threshold = 0.5
pred_train_onehot = (res_train_pb >= threshold).astype(int)
pred_test_onehot = (res_test_pb >= threshold).astype(int)
print('\nF1 \t micro \t macro')
print('train\t %.3f \t %.3f'%(f1_score(Y_train, pred_train_onehot, average='micro'), 
                           f1_score(Y_train, pred_train_onehot, average='macro')))
# print('test\t %.3f \t %.3f'%(f1_score(Y_test, pred_test_onehot, average='micro'), 
#                            f1_score(Y_test, pred_test_onehot, average='macro')))

train_max_pb = np.max(pred_train_pb, axis=0)
test_max_pb = np.max(pred_test_pb, axis=0)
train_sort = np.argsort(train_max_pb , axis=1)
test_sort = np.argsort(test_max_pb, axis=1)


train_sum = np.sum(pred_train_onehot,axis=1)
test_sum = np.sum(pred_test_onehot,axis=1)
for i in range(500):
    if train_sum[i] > 3:
        pred_train_onehot[i] = np.zeros(6)
        for j in [3,4,5]:
            pred_train_onehot[i][train_sort[i][j]] = 1

for i in range(66):
    if test_sum[i] > 3:
        pred_test_onehot[i] = np.zeros(6)
        for j in [3,4,5]:
            pred_test_onehot[i][test_sort[i][j]] = 1



print('\nF1 \t micro \t macro')
print('train\t %.3f \t %.3f'%(f1_score(Y_train, pred_train_onehot, average='micro'), 
                           f1_score(Y_train, pred_train_onehot, average='macro')))



# save results
np.save('test_one_hot', pred_test_onehot)
np.save('test_pb', res_test_pb)
np.savetxt('test_one_hot', pred_test_onehot, fmt='%i')
np.savetxt('test_pb', res_test_pb)

