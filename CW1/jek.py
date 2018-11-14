import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import timeit
import matplotlib as mpl

ds = sio.loadmat('face.mat')

X = ds['X']
l = ds['l'][0]
split = 0.8
W = 46
H = 56
IDs = 52
X = pd.DataFrame(X.transpose())
N = int(X.shape[0]*split)
l = pd.DataFrame(l)
X_train, X_test, l_train, l_test = train_test_split(X, l, test_size=(1-split), stratify = l)
# X_bar = np.mean(X, axis=1)
X_train, X_test = X_train.values, X_test.values
l_train, l_test = l_train.values, np.transpose(l_test.values)
X_bar = np.transpose([np.mean(X_train, axis=0)])
X_bar = np.ndarray.flatten(X_bar)
train_subspaces = []
for i in range(IDs):
    images = []
    for j in range(int(N)):
        if l_train[j][0] == i+1:
            images.append(X_train[j])
    train_subspaces.append(images)
eigen_subspaces = []
elements = np.asarray(train_subspaces[0])
elements_mean = np.mean(elements, axis=0)
elements = np.subtract(elements, elements_mean)
S_W = np.matmul(elements.T, elements)
mean_diff = np.atleast_2d(elements_mean)-X_bar
S_B = np.matmul(mean_diff.T, mean_diff)
S = np.matmul(elements, np.transpose(elements))
w, v = np.linalg.eig(S)
U = preprocessing.normalize(np.matmul(np.transpose(elements), v), axis=0)
eigen_subspaces.append([U, elements_mean])
for i in range(1, IDs):
    elements = np.asarray(train_subspaces[i])
    elements_mean = np.mean(elements, axis=0)
    elements = np.subtract(elements, elements_mean)
    S_W += np.matmul(elements.T, elements)
    mean_diff = np.atleast_2d(elements_mean)-X_bar
    S_B += np.matmul(mean_diff.T, mean_diff)
    S = np.matmul(elements, np.transpose(elements))
    w, v = np.linalg.eig(S)
    U = preprocessing.normalize(np.matmul(np.transpose(elements), v), axis=0)
    eigen_subspaces.append([U, elements_mean])

S_WB = np.matmul(np.linalg.inv(S_W), S_B)
w_lda, v_lda = np.linalg.eig(S_WB)

inds = w_lda.argsort()[::-1]
w_lda = w_lda[inds]
v_lda = v_lda[:, inds]
e = []
for M_lda in range(1,N+1):
    print(M_lda)
    w_lda = w_lda[:M_lda]
    v_lda = v_lda[:, :M_lda]

    A = np.subtract(X_train, X_bar).T
    #print(np.matmul(A.T, A).shape)
    w_pca, v_pca = np.linalg.eig((1/N)*np.matmul(A.T, A))
    inds = w_pca.argsort()[::-1]
    w_pca = w_pca[inds]
    v_pca = v_pca[:, inds]

    cum_w = np.cumsum(w_pca)/np.sum(w_pca)
    M_pca = np.argmax(cum_w > 0.95)
    w_pca = w_pca[:M_pca]
    v_pca = v_pca[:, :M_pca]
    v_pca = preprocessing.normalize(np.matmul(A, v_pca), axis=0)

    meh2 = np.dot(np.dot(v_pca.T, S_W), v_pca)
    meh1 = np.dot(np.dot(v_pca.T, S_B), v_pca)
    w_fld, v_fld = np.linalg.eig(np.linalg.inv(meh2).dot(meh1))
    w_fld = w_fld[:M_lda]
    v_fld = v_fld[:, :M_lda]

    v_opt = np.real(np.dot(v_pca, v_fld))
    v_opt = preprocessing.normalize(v_opt, axis=0)
    #print(np.transpose(v_opt)[0].shape)
    train_img = np.transpose(np.array_split(np.transpose(v_opt)[0], W))


    Y_train = []
    #print(A.shape)
    for i in range(int(N)):
        Y_train.append(np.matmul(v_opt.T, X_train[i]))


    correct = 0
    # X_test_norm = np.subtract(X_test,X_bar)
    X_test_norm = X_test
    for i in range(len(X_test)):
        y_test = np.matmul(v_opt.T, X_test_norm[i])
        diff = np.subtract(Y_train, y_test)
        value = np.argmin(np.diag(np.matmul(diff, diff.T)))
        if l_train[value] == l_test[0][i]:
            correct += 1
    print(correct/len(l_test[0]))
    e.append(correct/len(l_test[0]))

plt.plot(e)
plt.show()
