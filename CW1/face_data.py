import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# imports the dataset
ds = sio.loadmat('face.mat')
#X is the image data where each vector is an image
X = ds['X']
# l is a vector of image id where each ID states who the image is of
l = ds['l'][0]
#converts to pandas dataframes to allow easy spliting of data for training
X = pd.DataFrame(X.transpose())
N = X.shape[0]*0.75
l = pd.DataFrame(l)
X_train, X_test, l_train, l_test = train_test_split(X, l, test_size=0.25, random_state=42)
# X_bar = np.mean(X, axis=1)
X_train, X_test = X_train.values, X_test.values
# print(X_train.head())
# print(l_train.head())

X_bar = np.mean(X_train, axis=0)
# print(X_bar)
i = 0
for x in X_train:
    norm = [x - X_bar]
    norm = np.asarray(norm, dtype='float32')
    norm_T = np.transpose(norm)
    if i == 0:
        offset = np.dot(norm_T, norm)
        i += 1
        # print(norm.transpose())
    else:
        offset += np.dot(norm_T, norm)
    #     offset += np.matmul (norm.transpose(), norm)
offset = offset/N
# print(offset)
w, v = np.linalg.eig(offset)
# y = np.arange(w.shape[0])
sorted(w,reverse=True)
# plt.plot(w)
# plt.ylabel('Eigenvalue')
# plt.show()
aic = np.zeros_like(w)
N_eig = w.shape[0]
for i in range(N_eig):
    aic[i] = np.log10(w[i])+2*i/N
plt.plot(aic)
plt.show()
for i in range(N_eig):
    if (aic[i] - aic[i+1]) < 0:
        print(i)
        break
