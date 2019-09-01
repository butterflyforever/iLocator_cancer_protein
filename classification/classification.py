
# coding: utf-8

# In[ ]:

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
from sklearn.model_selection import GridSearchCV
from sklearn import manifold
from sklearn.metrics import classification_report


# # Read Featrues

# ## Patch feature

# In[ ]:

F = np.zeros((270, 500, 19))


# In[ ]:

# One image(3k x 3k) owns 90 patches(100x100)
two_imgs_list = []
six_imgs_list = []
size = label['Label'].values.size
for i in range(size):
    fname = label['ENSGID'].values[i].encode('ascii','ignore').strip(string.punctuation)+'.txt'
    tmp = np.loadtxt('patch_features/'+fname).astype(np.float)
    for j in range(180):
        F[j][i] = tmp[j]
    if tmp.shape[0] < 270 :
        two_imgs_list.append(i)
    else:
        for j in range(180, 270):
            F[j][i] = tmp[j]
    if tmp.shape[0] == 540:
        six_imgs_list.append(i)


# In[ ]:

for i in six_imgs_list:
    if i%2 == 1:
        fname = label['ENSGID'].values[i].encode('ascii','ignore').strip(string.punctuation)+'.txt'
        tmp = np.loadtxt('patch_features/'+fname).astype(np.float)
        for j in range(270, 540):
            F[j-270][i] = tmp[j]


# In[ ]:

# save np.array
np.save('./npy/F', F)


# ## Whole Feature

# In[ ]:

F1 = np.zeros((1,19))
F2 = np.zeros((1,19))
F3 = np.zeros((1,19))
two_imgs_list = []
six_imgs_list = []
size = label['Label'].values.size
for i in range(size):
    fname = label['ENSGID'].values[i].encode('ascii','ignore').strip(string.punctuation)+'.txt'
    tmp = np.loadtxt('patch_features/'+fname).astype(np.float)
    F1 = np.concatenate((F1, tmp[0].reshape(1,-1)), axis=0)
    F2 = np.concatenate((F2, tmp[1].reshape(1,-1)), axis=0)
    if tmp.shape[0] == 2:
        F3 = np.concatenate((F3, np.zeros((1,19))), axis=0)
        two_imgs_list.append(i)
    else:
        F3 = np.concatenate((F3, tmp[2].reshape(1,-1)), axis=0)
    if tmp.shape[0] == 6:
        six_imgs_list.append(i)
F1 = F1[1:, :]
F2 = F2[1:, :]
F3 = F3[1:, :]

# save np.array
np.save('F1', F1)
np.save('F2', F2)
np.save('F3', F3)
np.save('label', final_enc_label)


# In[ ]:

# save np.array
np.save('./npy/F1', F1)
np.save('./npy/F2', F2)
np.save('./npy/F3', F3)


# ## Matlab Feature

# In[ ]:

num_feature = 1097
MF1, MF2, MF3 = np.zeros((1,num_feature)), np.zeros((1,num_feature)), np.zeros((1,num_feature))
two_imgs_list = []
six_imgs_list = []
size = label['Label'].values.size
for i in range(size):
    fname = label['ENSGID'].values[i].encode('ascii','ignore').strip(string.punctuation)+'.txt'
    tmp = np.loadtxt('matlabFeature/'+fname).astype(np.float)
    MF1 = np.concatenate((MF1, tmp[0].reshape(1,-1)), axis=0)
    MF2 = np.concatenate((MF2, tmp[1].reshape(1,-1)), axis=0)
    if tmp.shape[0] == 2:
        MF3 = np.concatenate((MF3, np.zeros((1,num_feature))), axis=0)
        two_imgs_list.append(i)
    else:
        MF3 = np.concatenate((MF3, tmp[2].reshape(1,-1)), axis=0)
    if tmp.shape[0] == 6:
        six_imgs_list.append(i)
MF1 = MF1[1:, :]
MF2 = MF2[1:, :]
MF3 = MF3[1:, :]


# In[ ]:

for i in six_imgs_list:
    if i%2 == 1:
        fname = label['ENSGID'].values[i].encode('ascii','ignore').strip(string.punctuation)+'.txt'
        tmp = np.loadtxt('matlabFeature/'+fname).astype(np.float)
        MF1[i] = tmp[3]
        MF2[i] = tmp[4]
        MF3[i] = tmp[5]


# In[ ]:

# save np.array
np.save('./npy/MF1', MF1)
np.save('./npy/MF2', MF2)
np.save('./npy/MF3', MF3)


# # Load *.npy

# ## Matlab feature

# In[ ]:

MF1 = preprocessing.scale(np.load('./npy/MF1.npy'))
MF2 = preprocessing.scale(np.load('./npy/MF2.npy'))
MF3 = preprocessing.scale(np.load('./npy/MF3.npy'))
label = np.load('./npy/label.npy')[:,1:]
MF = np.concatenate((MF1, MF2, MF3), axis=0)


# In[ ]:

MF1.shape


# ### dimension reduction - PCA

# In[ ]:

# PCA
pca = PCA(random_state=0)
pca.fit(MF)
num_pc = 100
MF1_pca = pca.transform(MF1)[:, :num_pc]
MF2_pca = pca.transform(MF2)[:, :num_pc]
MF3_pca = pca.transform(MF3)[:, :num_pc]


# In[ ]:

MF1_pca.shape


# In[ ]:

# # Grid Search CV

# X_train, X_test, Y_train, Y_test = train_test_split(MF1_pca, label[:,0], 
#                                                     test_size=.2, random_state=4)

# tuned_parameters = [{'C':[0.01, 0.1, 1, 10]}]
# scores = ['f1_micro', 'f1_macro']
# # scores = ['recall', 'precision']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)

#     clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5, scoring='%s_macro' % score)
#     clf.fit(X_train, Y_train)

#     print("Best parameters set found on development set:")
#     print(clf.best_params_)
#     print("Grid scores on development set:")
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))

#     print("Detailed classification report:")
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     y_true, y_pred = Y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))


# In[ ]:

# train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(MF1_pca, label, test_size=.2, random_state=4)

acc_train, acc_test = [], []
rec_train, rec_test = [], []
pre_train, pre_test = [], []
pred_train_, pred_test_ = [], []

# classfiers
classifiers = {'svm':SVC(kernel='linear', verbose=True, probability=False, 
                         random_state=0, C=10),
              'LR':LogisticRegression(penalty='l1', C=1, random_state=0),
              'RF': RandomForestClassifier(n_estimators=10, criterion='gini', n_jobs=-1),
              'lda': LinearDiscriminantAnalysis()}

# train and test
for i in range(6):
    clf = classifiers['LR']
    clf.fit(X_train, Y_train[:,i])
    pred_train_pb = clf.predict_proba(X_train)
    pred_test_pb = clf.predict_proba(X_test)
#     pred_train = np.argmax(pred_train_pb, axis=1)
#     pred_test = np.argmax(pred_test_pb, axis=1)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    pred_train_.append(pred_train)
    pred_test_.append(pred_test)
    # accuracy
    acc_train.append(accuracy_score(Y_train[:,i], pred_train))
    acc_test.append(accuracy_score(Y_test[:,i], pred_test))
    # recall
    rec_train.append(metrics.recall_score(Y_train[:,i], pred_train))
    rec_test.append(metrics.recall_score(Y_test[:,i], pred_test))
    # precision
    pre_train.append(metrics.precision_score(Y_train[:,i], pred_train))
    pre_test.append(metrics.precision_score(Y_test[:,i], pred_test))

# evaluation
pred_test_ = np.array(pred_test_)
pred_train_ = np.array(pred_train_)
print('\nF1 \t micro \t macro')
print('train\t %.3f \t %.3f'%(f1_score(Y_train, np.transpose(pred_train_), average='micro'), 
                           f1_score(Y_train, np.transpose(pred_train_), average='macro')))
print('test\t %.3f \t %.3f'%(f1_score(Y_test, np.transpose(pred_test_), average='micro'),
                           f1_score(Y_test, np.transpose(pred_test_), average='macro')))


# In[ ]:

# # evaluation

# print('accuracy_train:')
# print(acc_train)

# print('accuracy_test:')
# print(acc_test)

# print('\n recall_train:')
# print(rec_train)

# print('recall_test:')
# print(rec_test)

# print('\n precision_train:')
# print(pre_train)

# print('precision_test:')
# print(pre_test)

# print('\n F1_micro')
# print('train %.3f'%f1_score(Y_train, np.transpose(pred_train_), average='micro'))
# print('test %.3f'%f1_score(Y_test, np.transpose(pred_test_), average='micro'))

# print('\n F1_macro')
# print('train %.3f'%f1_score(Y_train, np.transpose(pred_train_), average='macro'))
# print('test %.3f'%f1_score(Y_test, np.transpose(pred_test_), average='macro'))


# ### dimension reduction - manifold learning

# In[ ]:

X_train, X_test, Y_train, Y_test = train_test_split(MF1, label, test_size=.2, random_state=4)


# In[ ]:

# Isomap
isomap = manifold.Isomap(n_neighbors=300, n_jobs=-1, n_components=100)
isomap.fit(MF1)
X_train_new = isomap.transform(X_train)
X_test_new = isomap.transform(X_test)


# In[ ]:

# LLE
lle = manifold.LocallyLinearEmbedding(n_jobs=-1, n_components=100, n_neighbors=10)
lle.fit(MF1)
X_train_new = lle.transform(X_train)
X_test_new = lle.transform(X_test)


# In[ ]:

# t-SNE
tsne = manifold.TSNE(n_components=3, perplexity=20, learning_rate=200, verbose=1)
tsne.fit(MF1)
X_train_new = lle.transform(X_train)
X_test_new = lle.transform(X_test)


# In[ ]:

acc_train, acc_test = [], []
rec_train, rec_test = [], []
pre_train, pre_test = [], []
pred_train_, pred_test_ = [], []

# classfiers
classifiers = {'svm':SVC(kernel='linear', verbose=True, probability=False, 
                         random_state=0, C=10),
              'LR':LogisticRegression(penalty='l1', C=1, random_state=0),
              'RF': RandomForestClassifier(n_estimators=10, criterion='gini', n_jobs=-1),
              'lda': LinearDiscriminantAnalysis()}

# train and test
for i in range(6):
    clf = classifiers['LR']
    clf.fit(X_train_new, Y_train[:,i])
#     pred_train_pb = clf.predict_proba(X_train_new)
#     pred_test_pb = clf.predict_proba(X_test_new)
#     pred_train = np.argmax(pred_train_pb, axis=1)
#     pred_test = np.argmax(pred_test_pb, axis=1)
    pred_train = clf.predict(X_train_new)
    pred_test = clf.predict(X_test_new)
    pred_train_.append(pred_train)
    pred_test_.append(pred_test)
    # accuracy
    acc_train.append(accuracy_score(Y_train[:,i], pred_train))
    acc_test.append(accuracy_score(Y_test[:,i], pred_test))
    # recall
    rec_train.append(metrics.recall_score(Y_train[:,i], pred_train))
    rec_test.append(metrics.recall_score(Y_test[:,i], pred_test))
    # precision
    pre_train.append(metrics.precision_score(Y_train[:,i], pred_train))
    pre_test.append(metrics.precision_score(Y_test[:,i], pred_test))

# evaluation
pred_test_ = np.array(pred_test_)
pred_train_ = np.array(pred_train_)
print('\nF1 \t micro \t macro')
print('train\t %.3f \t %.3f'%(f1_score(Y_train, np.transpose(pred_train_), average='micro'), 
                           f1_score(Y_train, np.transpose(pred_train_), average='macro')))
print('test\t %.3f \t %.3f'%(f1_score(Y_test, np.transpose(pred_test_), average='micro'),
                           f1_score(Y_test, np.transpose(pred_test_), average='macro')))


# ### feature selection - Mutual Information

# In[ ]:

# train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(MF1, label, test_size=.2, random_state=4)

acc_train, acc_test = [], []
rec_train, rec_test = [], []
pre_train, pre_test = [], []
pred_train_, pred_test_ = [], []

# classfiers
classifiers = {'svm':SVC(kernel='linear', verbose=True, probability=False, 
                         random_state=0, C=10),
              'LR':LogisticRegression(penalty='l2', C=0.1, random_state=0),
              'RF': RandomForestClassifier(n_estimators=10, criterion='gini', n_jobs=-1)}

# train and test
for i in range(6):
    print('%d...'%(i+1))
    sel_k = SelectKBest(mutual_info_classif, k=100).fit(X_train, Y_train[:,i])
    X_train_new = sel_k.transform(X_train)
    X_test_new = sel_k.transform(X_test)
    clf = classifiers['LR']
    clf.fit(X_train_new, Y_train[:,i])
#     pred_train_pb = svm.predict_proba(X_train)
#     pred_test_pb = svm.predict_proba(X_test)
#     pred_train = np.argmax(pred_train_pb, axis=1)
#     pred_test = np.argmax(pred_test_pb, axis=1)
    pred_train = clf.predict(X_train_new)
    pred_test = clf.predict(X_test_new)
    pred_train_.append(pred_train)
    pred_test_.append(pred_test)
    # accuracy
    acc_train.append(accuracy_score(Y_train[:,i], pred_train))
    acc_test.append(accuracy_score(Y_test[:,i], pred_test))
    # recall
    rec_train.append(metrics.recall_score(Y_train[:,i], pred_train))
    rec_test.append(metrics.recall_score(Y_test[:,i], pred_test))
    # precision
    pre_train.append(metrics.precision_score(Y_train[:,i], pred_train))
    pre_test.append(metrics.precision_score(Y_test[:,i], pred_test))

# evaluation
pred_test_ = np.array(pred_test_)
pred_train_ = np.array(pred_train_)
print('\nF1 \t micro \t macro')
print('train\t %.3f \t %.3f'%(f1_score(Y_train, np.transpose(pred_train_), average='micro'), 
                           f1_score(Y_train, np.transpose(pred_train_), average='macro')))
print('test\t %.3f \t %.3f'%(f1_score(Y_test, np.transpose(pred_test_), average='micro'),
                           f1_score(Y_test, np.transpose(pred_test_), average='macro')))


# ## Patch feature

# In[ ]:

F = np.load('./npy/F.npy')
label = np.load('./npy/label.npy')[:,1:]
for i in range(180):
    F[i] = preprocessing.scale(F[i])


# In[ ]:

# acc_train, acc_test = [], []
# rec_train, rec_test = [], []
# pre_train, pre_test = [], []
res_train, res_test = [], []
svm = SVC(kernel='poly', verbose=True, probability=True, random_state=0, degree=3)
for i in range(6):
    X_train, Y_train = F[1][train_idx], label[train_idx][:,i]
    X_test, Y_test = F[1][test_idx], label[test_idx][:,i]
    svm.fit(X_train, Y_train)
#     pred_train_pb = svm.predict_proba(X_train)
#     pred_test_pb = svm.predict_proba(X_test)
#     pred_train = np.argmax(pred_train_pb, axis=1)
#     pred_test = np.argmax(pred_test_pb, axis=1)
    pred_train_ = svm.predict(X_train)
    pred_test_ = svm.predict(X_test)
    res_train.append(pred_train_)
    res_test.append(pred_test_)
#     # accuracy
#     acc_train.append(accuracy_score(Y_train, pred_train))
#     acc_test.append(accuracy_score(Y_test, pred_test))
#     # recall
#     rec_train.append(metrics.recall_score(Y_train, pred_train))
#     rec_test.append(metrics.recall_score(Y_test, pred_test))
#     # precision
#     pre_train.append(metrics.precision_score(Y_train, pred_train))
#     pre_test.append(metrics.precision_score(Y_test, pred_test))


# In[ ]:

# train_test_split
np.random.seed(36)
permute = np.random.permutation(np.arange(500))
train_idx = permute[:400]
test_idx = permute[400:]

X_train = np.zeros((1,19))
num_patch = 180
for patch in range(num_patch):
    X_train = np.concatenate((X_train, F[patch][train_idx]), axis=0)
X_train = X_train[1:,:]
Y_train = []
for i in range(6):
    tmp = np.repeat(label[train_idx][:, i], num_patch)
    Y_train.append(tmp)
Y_train = np.array(Y_train)
print(X_train.shape)
print(Y_train.shape)


# In[ ]:

RF = [RandomForestClassifier(verbose=False, n_jobs=-1, random_state=i) for i in range(6)]


# In[ ]:

# traing and predicting
num_patch = 180
num_class = 6
res_train, res_test = [], []
for i in range(num_class):
    print('fitting class %d'%i)
    RF[i].fit(X_train, Y_train[i])
    res_train_, res_test_ = [], []
    for patch in range(num_patch):
        if (patch+1)%10 == 0:
            print('\t predicting patch %d'%(patch+1))
        X_tra, Y_tra = F[patch][train_idx], label[train_idx][:,i]
        X_test, Y_test = F[patch][test_idx], label[test_idx][:,i]
        pred_train = RF[i].predict_proba(X_tra)
        pred_test = RF[i].predict_proba(X_test)
        res_train_.append(pred_train)
        res_test_.append(pred_test)
    res_train.append(res_train_)
    res_test.append(res_test_)
res_train = np.array(res_train)
res_test = np.array(res_test)
print(res_test.shape)
print(res_train.shape)


# In[ ]:

train_percent = []
test_percent = []
for i in range(6):
    train_percent.append(1.*np.sum(label[train_idx][:,i])/400)
    test_percent.append(1.*np.sum(label[test_idx][:,i])/100)


# In[ ]:

f_res_train = np.zeros(train.shape).astype(int)
f_res_test = np.zeros(test.shape).astype(int)
for i in range(6):
    # test
    test_true_idx = np.flip(np.argsort(test[i]), axis=0)[:int(round(train_percent[i]*100))]
    f_res_test[i][test_true_idx] = 1
    # train
    train_true_idx = np.flip(np.argsort(train[i]), axis=0)[:int(round(train_percent[i]*100))]
    f_res_train[i][train_true_idx] = 1


# In[ ]:

svm = SVC(kernel='poly', verbose=True, probability=True, random_state=0, degree=3)
num_patch = 180
res_train_, res_test_ = [], []
for patch in range(num_patch):
    print(patch)
    X_train, X_test = F[patch][train_idx], F[patch][test_idx]
    res_train, res_test = [], []
    for i in range(6):
        Y_train, Y_test = label[train_idx][:, i], label[test_idx][:, i]
        svm.fit(X_train, Y_train)
        pred_train = svm.predict(X_train)
        pred_test = svm.predict(X_test)
        res_train.append(pred_train_)
        res_test.append(pred_test_)
    res_train_.append(res_train)
    res_test_.append(res_test)
res_test_ = np.array(res_test_)
res_train_ = np.array(res_train_)


# In[ ]:

# br = OneVsRestClassifier(SVC(kernel='poly', probability=False, 
#                              random_state=0, degree=3, verbose=True),
#                          n_jobs=-1)
# br.fit(F[1][test_idx], label[test_idx])
# # pred_train_pb = br.predict_proba(F[1][train_idx])
# # pred_test_pb = br.predict_proba(F[1][test_idx])
# pred_train = br.predict(F[1][train_idx])
# pred_test = br.predict(F[1][test_idx])


# In[ ]:

# np.random.seed(4)
# permute = np.random.permutation(np.arange(500))
# train_idx = permute[:400]
# test_idx = permute[400:]


# In[ ]:

# X_train_1 = np.zeros((1,19))
# X_train_2 = np.zeros((1,19))
# X_train_3 = np.zeros((1,19))
# for i in train_idx:
#     X_train_1 = np.concatenate((X_train_1, PF1[i*90:(i+1)*90]), axis=0)
#     X_train_2 = np.concatenate((X_train_2, PF2[i*90:(i+1)*90]), axis=0)
#     X_train_3 = np.concatenate((X_train_3, PF3[i*90:(i+1)*90]), axis=0)
# X_train_1 = X_train_1[1:]
# X_train_2 = X_train_2[1:]
# X_train_3 = X_train_3[1:]


# In[ ]:

# X_test_1 = np.zeros((1,19))
# X_test_2 = np.zeros((1,19))
# X_test_3 = np.zeros((1,19))
# for i in test_idx:
#     X_test_1 = np.concatenate((X_test_1, PF1[i*90:(i+1)*90]), axis=0)
#     X_test_2 = np.concatenate((X_test_2, PF2[i*90:(i+1)*90]), axis=0)
#     X_test_3 = np.concatenate((X_test_3, PF3[i*90:(i+1)*90]), axis=0)
# X_test_1 = X_test_1[1:]
# X_test_2 = X_test_2[1:]
# X_test_3 = X_test_3[1:]


# In[ ]:

# Y_train = np.zeros((1,6)).astype(int)
# for i in train_idx:
#     Y_train = np.concatenate((Y_train, np.repeat(label[i:i+1], 90, axis=0)), axis=0)
# Y_train = Y_train[1:]


# In[ ]:

# Y_test = label[test_idx]


# In[ ]:

# for i in two_imgs_list:
#     if i in train_idx:
# #         print(i)
#         idx = np.argwhere(train_idx==i)[0,0]
# #         print(X_train_3[idx*90:(idx+1)*90])
#     if i in test_idx:
# #         print(i)
#         idx = np.argwhere(test_idx==i)[0,0]
# #         print(X_test_3[idx*90:(idx+1)*90])


# ## Whole image feature

# In[ ]:

for i in range(6):
    print('class %d: %.2f '%(i, 100.*np.sum(Y_train[:,i])/Y_train[:,i].size))


# In[ ]:

F1 = preprocessing.scale(np.load('./npy/F1.npy'))
F2 = preprocessing.scale(np.load('./npy/F2.npy'))
F3 = preprocessing.scale(np.load('./npy/F3.npy'))
label = np.load('./npy/label.npy')[:,1:]


# ### Use F1

# In[ ]:

X_train, X_test, Y_train, Y_test = train_test_split(F1, label, test_size=.2, random_state=4)
acc_train, acc_test = [], []
rec_train, rec_test = [], []
pre_train, pre_test = [], []
for i in range(6):
    svm = SVC(kernel='poly', verbose=True, probability=True, random_state=0)
    svm.fit(X_train, Y_train[:,i])
    pred_train_pb = svm.predict_proba(X_train)
    pred_test_pb = svm.predict_proba(X_test)
    pred_train = np.argmax(pred_train_pb, axis=1)
    pred_test = np.argmax(pred_test_pb, axis=1)
    # accuracy
    acc_train.append(accuracy_score(Y_train[:,i], pred_train))
    acc_test.append(accuracy_score(Y_test[:,i], pred_test))
    # recall
    rec_train.append(metrics.recall_score(Y_train[:,i], pred_train))
    rec_test.append(metrics.recall_score(Y_test[:,i], pred_test))
    # precision
    pre_train.append(metrics.precision_score(Y_train[:,i], pred_train))
    pre_test.append(metrics.precision_score(Y_test[:,i], pred_test))


# In[ ]:

print('accuracy_train:')
print(acc_train)

print('accuracy_test:')
print(acc_test)

print('\nrecall_train:')
print(rec_train)

print('recall_test:')
print(rec_test)

print('\nprecision_train:')
print(pre_train)

print('precision_test:')
print(pre_test)


# ### Use F2

# In[ ]:

X_train, X_test, Y_train, Y_test = train_test_split(F2, label, test_size=.2, random_state=4)
acc_train, acc_test = [], []
for i in range(6):
    svm = SVC(kernel='poly', verbose=True, probability=True, random_state=0)
    svm.fit(X_train, Y_train[:,i])
    pred_train = svm.predict(X_train)
    pred_test = svm.predict(X_test)
    acc_train.append(accuracy_score(pred_train, Y_train[:,i]))
    acc_test.append(accuracy_score(pred_test, Y_test[:,i])) 


# In[ ]:

print(acc_test)
print(acc_train)


# ### Use F3

# In[ ]:

X_train, X_test, Y_train, Y_test = train_test_split(F3, label, test_size=.2, random_state=4)
acc_train, acc_test = [], []
for i in range(6):
    svm = SVC(kernel='poly', verbose=True, probability=True, random_state=0)
    svm.fit(X_train, Y_train[:,i])
    pred_train = svm.predict(X_train)
    pred_test = svm.predict(X_test)
    acc_train.append(accuracy_score(pred_train, Y_train[:,i]))
    acc_test.append(accuracy_score(pred_test, Y_test[:,i])) 


# In[ ]:

print(acc_test)
print(acc_train)


# ### K-fold

# In[ ]:

kf = KFold(n_splits=5, random_state=4)


# In[ ]:

acc_train, acc_test = [], []
for train_index, test_index in kf.split(F2):
    X_train, X_test = F2[train_index], F2[test_index]
    Y_train, Y_test = label[train_index], label[test_index]
    acc_tr, acc_te = [], []
    for i in range(6):
        svm = SVC(kernel='rbf', verbose=True, probability=True, random_state=0, degree=5)
        svm.fit(X_train, Y_train[:,i])
        pred_train = svm.predict(X_train)
        pred_test = svm.predict(X_test)
        acc_tr.append(accuracy_score(pred_train, Y_train[:,i]))
        acc_te.append(accuracy_score(pred_test, Y_test[:,i]))
    acc_train.append(acc_tr)
    acc_test.append(acc_te)
acc_test = np.array(acc_test)
acc_train = np.array(acc_train)


# In[ ]:

acc_train


# In[ ]:

acc_test


# In[ ]:

np.mean(acc_test, axis=0)


# # CC

# In[ ]:

chains = [ClassifierChain(SVC(kernel='poly', probability=True, random_state=0, verbose=True), 
                          order='random', random_state=i) for i in range(10)]


# In[ ]:

for i, chain in enumerate(chains):
    print('\nchain %d: '%i)
    chain.fit(X_train, Y_train)


# In[ ]:

pred_train_chains_pb = np.array([chain.predict_proba(X_train) for chain in chains])
pred_train_ensemble_pb = pred_train_chains_pb.mean(axis=0)
pred_train_ens = np.argmax(pred_train_ensemble_pb, axis=1)

pred_test_chains_pb = np.array([chain.predict_proba(X_test) for chain in chains])
pred_test_ensemble_pb = pred_test_chains_pb.mean(axis=0)
pred_test_ens = np.argmax(pred_test_ensemble_pb, axis=1)


# In[ ]:

Y_train


# In[ ]:

pred_train_ens


# In[ ]:

# print(f1_score(Y_train, pred_train_ens, average='macro'))
# print(f1_score(Y_train, pred_train_ens, average='micro'))
# print(f1_score(Y_test, pred_test_ens, average='macro'))
# print(f1_score(Y_test, pred_test_ens, average='micro'))


# # BR

# In[ ]:

br = OneVsRestClassifier(SVC(kernel='poly', probability=True, random_state=0, degree=4))
br.fit(X_train, Y_train)
pred_train_pb = br.predict_proba(X_train)
pred_test_pb = br.predict_proba(X_test)
pred_train = br.predict(X_train)
pred_test = br.predict(X_test)


# In[ ]:

print(f1_score(Y_train, pred_train, average='macro'))
print(f1_score(Y_train, pred_train, average='micro'))
print(f1_score(Y_test, pred_test, average='macro'))
print(f1_score(Y_test, pred_test, average='micro'))


# In[ ]:

br = OneVsRestClassifier(SVC(kernel='linear'))
br.fit(X_train, Y_train)
pred_train = br.predict(X_train)
pred_test = br.predict(X_test)
print(f1_score(Y_train, pred_train, average='macro'))
print(f1_score(Y_train, pred_train, average='micro'))
print(f1_score(Y_test, pred_test, average='macro'))
print(f1_score(Y_test, pred_test, average='micro'))


# In[ ]:

br = OneVsRestClassifier(LogisticRegression())
br.fit(X_train, Y_train)
pred_train = br.predict(X_train)
pred_test = br.predict(X_test)
print(f1_score(Y_train, pred_train, average='macro'))
print(f1_score(Y_train, pred_train, average='micro'))
print(f1_score(Y_test, pred_test, average='macro'))
print(f1_score(Y_test, pred_test, average='micro'))

