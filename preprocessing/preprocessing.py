
# coding: utf-8

# #### encodes labels of samples

# In[1]:

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


# # Encode Label



label = pd.read_excel('data_label.xlsx')


sp = string.punctuation.replace('(','').replace(')','')
size = label['Label'].values.size
cate = []
for i in range(size):
    a = label['Label'].values[i]
    s = a.encode('ascii','ignore')
    s = s.strip(sp).split(';')
    cate = list(set().union(s,cate))



print('\n# of categories: %d'%len(cate))

le = preprocessing.LabelEncoder()
le.fit(cate)

enc_label = []
for i in range(size):
    a = label['Label'].values[i]
    s = a.encode('ascii','ignore').strip(sp).split(';')
    s_ = list(le.transform(s))
    enc_label.append(s_)
enc_label = MultiLabelBinarizer().fit_transform(enc_label)


print(enc_label)
enc_label.shape

print(le.transform(['', 'Cytoplasm','Mitochondria','Vesicles', 'Endoplasmic reticulum', 'Golgi apparatus']))
print(le.transform(['Nucleus but not nucleoli','Nucleoli','Nucleus']))

label_list = le.transform(['', 'Cytoplasm','Mitochondria', 'Nucleus','Nucleoli','Nucleus but not nucleoli', 
              'Vesicles', 'Endoplasmic reticulum', 'Golgi apparatus'])
print(label_list)

enc_label_ = enc_label[:,label_list]
final_enc_label = np.concatenate((enc_label_[:,:3], 
                                  (np.sum(enc_label_[:, [3,4,5]], axis=1)>0).astype(int).reshape(-1,1), 
                                  enc_label_[:,6:]), axis=1)
# np.save('label', final_enc_label)


# Final Labels

# None: 0 
# Cytoplasm: 1
# Mitochondria: 2 
# Nucleus, Nucleoli, Nucleus but not nucleoli: 3
# Vesicles: 4
# Endoplasmic reticulum: 5 
# Golgi apparatus: 6


# # Statistics


print('Number of labels')
print('0 labels: %d samples'%np.sum(enc_label[:,0]==1))
print('1 labels: %d samples'%np.sum(np.sum(enc_label[enc_label[:,0]==0],axis=1)==1))
print('2 labels: %d samples'%np.sum(np.sum(enc_label[enc_label[:,0]==0],axis=1)==2))
print('3 labels: %d samples'%np.sum(np.sum(enc_label[enc_label[:,0]==0],axis=1)==3))



L = [133,274,82,11]
plt.bar(np.arange(4), L)
for i in range(4):
    plt.text(i-0.1, L[i]+1, L[i])
plt.ylabel('number of samples')
plt.xlabel('number of labels')
plt.xticks(np.arange(4))
plt.savefig('label_vs_samples.png', dpi=100)
plt.show()


print('Annotation Reliability')
print('supportive: %d samples'%np.sum(label['Annotation Reliability'].values == u"'Supportive'"))
print('uncertain: %d samples'%np.sum(label['Annotation Reliability'].values == u"'Uncertain'"))
print('unknown: %d samples'%np.sum(label['Annotation Reliability'].values == u'[]'))


label_num_sample = []
num_label = enc_label.shape[1]
for i in range(num_label):
    label_num_sample.append(np.sum(enc_label[:,i]))
print(label_num_sample)


plt.bar(np.arange(num_label), label_num_sample)
for i in range(num_label):
    plt.text(i-0.5, label_num_sample[i]+1, label_num_sample[i])
plt.ylabel('number of samples')
plt.xlabel('class')
plt.xticks(np.arange(num_label))
plt.savefig('class_vs_samples.png', dpi=100)
plt.show()

