import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import timeit
import matplotlib as mpl
import random

mpl.rcParams['figure.figsize'] = (10,15)
ds = sio.loadmat('face.mat')

X = ds['X']
l = ds['l'][0]
split = 0.8
W = 46
H = 56
IDs = 52
X = pd.DataFrame(X.transpose())
l = pd.DataFrame(l)

#bagging
bag_ratio = 0.75
M_pca_ratio = 0.95
M_lda_ratio = 0.1
N = round(X.shape[0]*split)
N_bag = round(X.shape[0]*split*bag_ratio) #different for each bag now

X_train, X_test, l_train, l_test = train_test_split(X, l, test_size=(1-split), stratify = l)
# X_bar = np.mean(X, axis=1)
X_train, X_test = X_train.values, X_test.values
l_train, l_test = l_train.values, np.transpose(l_test.values)
X_bar = np.transpose([np.mean(X_train, axis=0)])
X_bar = np.ndarray.flatten(X_bar)
print('    l_train.shape',l_train.shape)
print('    X_train.shape', X_train.shape)

def bag_train_space(X_bag_data, l_bag_data, ratio):

    train_subspaces = []

    for i in range(IDs):
        images = []
        for j in range(round(N*ratio)):
            if l_bag_data[j] == i+1:
                images.append(X_bag_data[j])
        train_subspaces.append(images)
    return np.asarray(train_subspaces)

def SWSB(bag_subspaces, X_bar_bag):
    elements = np.asarray(bag_subspaces[0])
    elements_mean = np.mean(elements, axis=0)
    elements = np.subtract(elements,elements_mean)
    S_W = np.matmul(elements.T,elements)
    mean_diff = np.atleast_2d(elements_mean)-X_bar_bag
#     print(mean_diff.shape)
    S_B = np.matmul(mean_diff.T, mean_diff)
    for i in range(1,IDs):
#         print(i)
        elements = np.asarray(bag_subspaces[i])
#         print(train_subspaces.shape)
        elements_mean = np.mean(elements, axis=0)
        elements = np.subtract(elements,elements_mean)
        S_W += np.matmul(elements.T,elements)
        mean_diff = np.atleast_2d(elements_mean)-X_bar_bag
        S_B += np.matmul(mean_diff.T, mean_diff)

    return S_W, S_B


#PCA
def rand_PCA(X_train_pca, X_bar_pca):
    A = np.subtract(X_train_pca,X_bar_pca).T

    w_pca, v_pca = np.linalg.eig((1/N)*np.matmul(A.T,A))
    inds = w_pca.argsort()[::-1]

    w_pca = w_pca[inds]
    v_pca = v_pca[:,inds]

    M_0 = 60
    M_1 = 40

    mask = []

    for p in range(0,M_1):
        x = random.randint(M_0,415)
        while x in mask:
            x = random.randint(M_0,415)
        mask.append(x)

    w_pca_0 = w_pca[:M_0]
    v_pca_0 = v_pca[:,:M_0]

    w_pca_1 = w_pca[mask]
    v_pca_1 = v_pca[:,mask]


    v_pca_full = np.append(v_pca_0.T, v_pca_1.T, axis=0)

    v_pca_full = preprocessing.normalize(np.matmul(A,v_pca_full.T), axis=0)
    w_pca_full = np.append(w_pca_0, w_pca_1)


    return w_pca_full, v_pca_full


#LDA
def LDA(S_W_bag, S_B_bag, v_pca):
    proj_1 = np.dot(np.dot(v_pca.T,S_W_bag),v_pca)
    proj_2 = np.dot(np.dot(v_pca.T,S_B_bag),v_pca)

    w_fld, v_fld = np.linalg.eig(np.linalg.inv(proj_1).dot(proj_2))

    cum_w = np.cumsum(w_fld)/np.sum(w_fld);
    M_lda = 35
#     print('    M LDA', M_lda)

    inds = w_fld.argsort()[::-1]
    w_fld = w_fld[inds]
#     print(w_fld.shape)
    v_fld = v_fld[:,inds]


    w_fld = w_fld[:M_lda]
    v_fld = v_fld[:,:M_lda]
    return w_fld, v_fld


#W_OPT
def w_opt(v_pca, v_fld):
    v_opt = np.dot(v_pca,v_fld)
    v_opt = preprocessing.normalize(np.real(v_opt),axis=0)
    return v_opt

#Get error of each bag
def error(v_opt, X_train_dataset, X_test_data, l_train_error, l_test_error):
    Y_train = []
    for i in range(int(N_bag)):
        Y_train.append(np.matmul(v_opt.T,X_train_dataset[i]))

    correct = 0
    l_pred = []

    for i in range(len(X_test)):
        y_test = np.matmul(v_opt.T, X_test_data[i])
        diff = np.subtract(Y_train, y_test)
        pred = np.diag(np.matmul(diff, diff.T))
        value = np.argmin(pred)
        l_pred.append(l_train_error[value])
        if l_train_error[value] == l_test_error[0][i]:
            correct += 1

    return correct/len(l_test[0]), l_pred

def N_space_error(No_models, X_train, l_train):
    model_error = []
    model_pred = []

    model_space = bag_train_space(X_train, l_train, 1)
#     print('    ', model_space.shape)
    S_W, S_B = SWSB(model_space, X_bar)
    for i in range(No_models):
        # print('    Model:', i+1)
        w_pca, v_pca = rand_PCA(X_train, X_bar)
#         plt.plot(w_pca)
        w_fld, v_fld = LDA(S_W, S_B, v_pca)
        v_opt = w_opt(v_pca, v_fld)
        err, l_pred = error(v_opt, X_train, X_test, l_train, l_test)
        # print('    ', err)
        model_error.append(err)
        model_pred.append(l_pred)


    return model_error, model_pred


def commit_cor(bags_pred, l_test):
    correct = 0
    for i in range(len(bags_pred)):
#     for i in range(bags_pred.shape[0]):
#         value = np.floor(np.argmin(bags_pred[i])/round(8*bag_ratio))+1
# #         print(value, l_test[0][i])
        if bags_pred[i] == l_test[0][i]:
            correct += 1
    return correct/104

def commit_mac(bags_pred, bags_err):

    y_pred = []
    for i in range(bags_pred.T.shape[0]):
        im_pred = np.zeros(IDs)
        for j in range(bags_pred.T.shape[1]):
            im_pred[bags_pred.T[i][j]-1] += 1
        y_pred.append(np.argmax(im_pred)+1)
    return y_pred



# models_error, models_pred = N_space_error(20, X_train, l_train)
#
# models_pred = np.asarray(models_pred)
# models_pred = np.squeeze(models_pred, axis=2)
# models_commit = commit_mac(models_pred, models_error)
#
# print('    ', commit_cor(models_commit, l_test))

# f = open('features.csv', 'a+')
data = []
from tqdm import tqdm
M_0_max = 120
M_MAX = 180
for i in tqdm(range(20,M_0_max, 10)):
    for j in tqdm(range(20,M_MAX - i, 10)):
        M_lda = 35
        M_0 = i
        M_1 = j
        models_error, models_pred = N_space_error(20, X_train, l_train)
        models_pred = np.asarray(models_pred)
        models_pred = np.squeeze(models_pred, axis=2)
        models_commit = commit_mac(models_pred, models_error)
        correct = commit_cor(models_commit, l_test)
        data.append([correct, M_0, M_1])
        print('    ', correct, M_0, M_1)

data = np.asarray(data)
np.savetxt("features2.csv", data, delimiter=",")
