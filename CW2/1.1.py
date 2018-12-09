
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.io import loadmat
import json
train_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()
camId = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()
filelist = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten()
gallery_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten()
labels = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()
query_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten()
with open('PR_data/feature_data.json', 'r') as f:
    features = np.array(json.load(f))
train_idx -= 1
gallery_idx -= 1
query_idx -= 1


# In[2]:


train_features = features[train_idx.tolist()]
gallery_features = features[gallery_idx.tolist()]
query_features = features[query_idx.tolist()]
gallery_label = labels[gallery_idx.tolist()]
query_label = labels[query_idx.tolist()]
gallery_cam = camId[gallery_idx.tolist()]
query_cam = camId[query_idx.tolist()]


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=15,metric='minkowski', p =3)
classifier.fit(gallery_features,gallery_label)
values = classifier.kneighbors(query_features)


# In[ ]:


rank1 = 0
rank5 = 0
rank10 = 0

for i in range(len(query_idx)):
    cam_match = 0
    found = False
    for j in range(len(values[1][i])):
        if gallery_label[values[1][i][j]] == query_label[i] and gallery_cam[values[1][i][j]] != query_cam[i] and found == False:
            if j - cam_match == 0:
                rank1 += 1
                rank5 += 1
                rank10 += 1
                found = True
            elif j - cam_match <= 4:
                rank5 += 1
                rank10 += 1
                found = True
            elif j - cam_match <= 9:
                rank10 += 1
                found = True
        elif gallery_label[values[1][i][j]] == query_label[i] and gallery_cam[values[1][i][j]] == query_cam[i]:
            cam_match += 1


# In[ ]:



print(rank1/len(query_label))
print(rank5/len(query_label))
print(rank10/len(query_label))

