
# coding: utf-8

# In[2]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# In[3]:


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
features = np.divide(features,np.amax(features))


# In[4]:


train_features = features[train_idx.tolist()]
gallery_features = features[gallery_idx.tolist()]
query_features = features[query_idx.tolist()]

train_label = labels[train_idx.tolist()]
gallery_label = labels[gallery_idx.tolist()]
query_label = labels[query_idx.tolist()]

train_cam = camId[train_idx.tolist()]
gallery_cam = camId[gallery_idx.tolist()]
query_cam = camId[query_idx.tolist()]

labeled_train = np.asarray(list(zip(train_features, train_label, train_cam)))
labeled_gallery = np.asarray(list(zip(gallery_features, gallery_label, gallery_cam)))
labeled_query = np.asarray(list(zip(query_features, query_label, query_cam)))


# ### Formalise training data into input triples and output pairs

# In[15]:
for query_i in range(10):
    import random as rand
    from more_itertools import locate
    input_shape = (3, 2048, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2),
                        activation='relu',
                        input_shape=input_shape))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

    batch_size = 128
    epochs = 20
    while(1):
        in_triples_train = np.empty((7368, 3), int)
        out_pairs_train =  np.empty((7368, 2), int)
        for i in range(labeled_train.shape[0]) :

            
            current_id = labeled_train[i][1]
            current_cam = labeled_train[i][2]
            #ensure there are three unique ID's randomly selected
            rand_id1 = rand.choice(train_label)
            rand_id2 = rand.choice(train_label)
            while not((rand_id1 != current_id) and (rand_id2 != current_id) and (rand_id1 != rand_id2)) :
                rand_id1 = rand.choice(train_label)
                rand_id2 = rand.choice(train_label)


            triple_set = list(locate(labeled_train, lambda x: 
                                (x[1] == current_id and x[2] != current_cam)  
                                or (x[1] == rand_id1)
                                or (x[1] == rand_id2)
                                    ))

            triple_1 = rand.choice(triple_set)
            triple_2 = rand.choice(triple_set)
            while triple_1 == triple_2 : 
                triple_1 = rand.choice(triple_set)
                triple_2 = rand.choice(triple_set)

            in_triples_train[i] = [triple_1, i, triple_2]

            out_pairs_train[i] = [(current_id == labeled_train[triple_1][1]),(current_id == labeled_train[triple_2][1])]


        # ### Formalise test data into input triple and output pairs

        # In[16]:


        x_train = np.empty((7368, 3, 2048), float)
        for i in range(in_triples_train.shape[0]):
            x_train[i] = train_features[in_triples_train[i]]
            

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

        y_train = out_pairs_train


        # ### Set up Neural Network

        # In[17]:



        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1)
        if(0.65 < model.history.history['acc'][-1]):
            break
    print(model.history.history['acc'][-1])
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


    # In[18]:
    model_name = 'model_' + str(model.history.history['acc'][-1])[2:6] + '.h5'

    model.save(model_name)


    # ``` python
    #     predict([gallery[i], [query], [gallery[!i])                           
    # ```
    # - ensure query[!i] does not use the same camera as test_data
    # - for each i generate highest probability test_data matches to it with all other queries !i (set to zero for invalid !i)
    # - sort the list of probabilities indexed for query[i]
    # - choose top *r* depending on rank
    # - gereate score 1 or 0 dependin on if the correct match is found within that rank
    # - repeat for all Test_data
    # 

    # Alternate prediction loop choosing a single K with a different label to I

    # In[8]:


    # from tqdm import tqdm_notebook as tqdm
    # predictions = np.zeros((1400, 5328, 3))

    # # for i in tqdm(range(labeled_gallery.shape[0])) :
    # for i in tqdm(range(predictions.shape[0])) :
    #     j_pred = np.zeros((5328, 3))
        
    #     k = rand.randrange(gallery_label.shape[0])

    #     for j in range(labeled_gallery.shape[0]):
    #         k_pred = 0
            
            
    #         if k == j: #check for same data point or same cam
    #             continue
    #         x_pred = np.empty((1, 3, 2048), float)
    #         x_pred[0][0] = gallery_features[j]
    #         x_pred[0][1] = query_features[i]
    #         x_pred[0][2] = gallery_features[k]
    #         x_pred = x_pred.reshape(1, 3, 2048, 1)
    #         k_pred = model.predict(x_pred)[0][0]
            
    # #         print(k_pred , labeled_query[j][1])
            
            
    #         j_pred[j] = [k_pred, labeled_gallery[j][1], labeled_gallery[j][2]] #store best prediction and its ID
            
    #     inds = np.argsort(j_pred.T[0])
    #     j_pred = np.asarray(list(zip(j_pred.T[0][inds], j_pred.T[1][inds], j_pred.T[2][inds])))
    # #     print(j_pred)
    #     predictions[i] = j_pred #sort list by descending probability


    #predictions should now be an sorted array of best predictions of each 
    #gallery image in descenting order in a list with the corresponding predicted ID


    # In[58]:


    from tqdm import tqdm
    predictions = np.zeros((1400, 5328, 3))

    # for i in tqdm(range(labeled_gallery.shape[0])) :
    for i in tqdm(range(1400)) :
        j_pred = np.zeros((5328, 3))
        
        k = rand.randrange(gallery_label.shape[0])

    
        x_pred = np.empty((3, 5328, 2048), float)
        x_pred[0]= gallery_features
        x_pred[1] = query_features[i]
        x_pred[2] = gallery_features[k]
        x_pred = np.transpose(x_pred,(1,0,2))
        x_pred = x_pred.reshape(gallery_features.shape[0], 3, 2048, 1)
        k_pred = model.predict(x_pred).T[0]

        j_pred = np.asarray([k_pred, gallery_label, gallery_cam]) #store predictions, label and camera

        inds = np.argsort(j_pred[0])
        j_pred = np.asarray(list(zip(j_pred[0][inds], j_pred[1][inds], j_pred[2][inds])))
    #     print(j_pred)
        predictions[i] = j_pred #sort list by descending probability


    # In[57]:




    # ##### Get the rank accuracies of the resulting predictions

    # In[59]:


    ranks = range(1,101) #set ranks in a list to check
    rank_scores = [0]*len(ranks)
    # for i in range(predictions.shape[0]) :
    for i in range(predictions.shape[0]) :
        cam_match = 0
        for j in range(predictions.shape[1]):
            if int(predictions[i][5327-j][1]) == labeled_query[i][1] and int(predictions[i][5327-j][2]) != labeled_query[i][2]:
                for k in range(len(ranks)):
                    if j - cam_match <= ranks[k]-1:
                        rank_scores[k] += 1
                break
            elif int(predictions[i][5327-j][1]) == labeled_query[i][1] and int(predictions[i][5327-j][2]) == labeled_query[i][2]:
                cam_match += 1

    for i in range(len(rank_scores)) :
        rank_scores[i] = rank_scores[i]/predictions.shape[0]
        # print('Rank: ', ranks[i], ' Accuracy: ', rank_scores[i])

    for i in [0,4,9]:
        print(i+1, '   ', rank_scores[i])
    # In[11]:


    # In[12]:

    pred_name = 'Prediction_'+str(model.history.history['acc'][-1])[2:6]
    ranks_name = 'Ranks_'+str(model.history.history['acc'][-1])[2:6]
    np.save(pred_name, predictions)
    np.save(ranks_name, rank_scores)
    keras.backend.clear_session()

