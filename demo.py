import argparse
import os
import pdb
import numpy as np
import math
import itertools
import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from torch_geometric.data import Data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv
from torch_geometric.nn import BatchNorm, EdgePooling, TopKPooling, global_add_pool
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.io
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import random
from gGAN import gGAN, netNorm

torch.cuda.empty_cache()
torch.cuda.empty_cache()

# random seed
manualSeed = 1

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('running on GPU')
    # if you are using GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

else:
    device = torch.device("cpu")
    print('running on CPU')


def demo():
    def cast_data(array_of_tensors, version):
        version1 = torch.tensor(version, dtype=torch.int)

        N_ROI = array_of_tensors[0].shape[0]
        CHANNELS = 1
        dataset = []
        edge_index = torch.zeros(2, N_ROI * N_ROI)
        edge_attr = torch.zeros(N_ROI * N_ROI, CHANNELS)
        x = torch.zeros((N_ROI, N_ROI))  # 35 x 35
        y = torch.zeros((1,))

        counter = 0
        for i in range(N_ROI):
            for j in range(N_ROI):
                edge_index[:, counter] = torch.tensor([i, j])
                counter += 1
        for mat in array_of_tensors:  # 1,35,35,4

            if version1 == 0:
                edge_attr = mat.view(1225, 1)
                x = mat.view(nbr_of_regions, nbr_of_regions)
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                x = torch.tensor(x, dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                dataset.append(data)

            elif version1 == 1:
                edge_attr = torch.randn(N_ROI * N_ROI, CHANNELS)
                x = torch.randn(N_ROI, N_ROI)  # 35 x 35
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                x = torch.tensor(x, dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                dataset.append(data)

        return dataset

    #####################################################################################################

    def linear_features(data):
        n_roi = data[0].shape[0]
        n_sub = data.shape[0]
        counter = 0

        num_feat = (n_roi * (n_roi - 1) // 2)
        final_data = np.empty([n_sub, num_feat], dtype=float)
        for k in range(n_sub):
            for i in range(n_roi):
                for j in range(i+1, n_roi):
                    final_data[k, counter] = data[k, i, j]
                    counter += 1
            counter = 0

        return final_data

    def make_sym_matrix(nbr_of_regions, feature_vector):
        sym_matrix = np.zeros([9, feature_vector.shape[1], nbr_of_regions, nbr_of_regions], dtype=np.double)
        for j in range(9):
            for i in range(feature_vector.shape[1]):
                my_matrix = np.zeros([nbr_of_regions, nbr_of_regions], dtype=np.double)

                my_matrix[np.triu_indices(nbr_of_regions, k=1)] = feature_vector[j, i, :]
                my_matrix = my_matrix + my_matrix.T
                my_matrix[np.diag_indices(nbr_of_regions)] = 0
                sym_matrix[j, i,:,:] = my_matrix

        return sym_matrix

    def plot_predictions(predicted, fold):
        plt.clf()
        for j in range(predicted.shape[0]):
            for i in range(predicted.shape[1]):
                predicted_sub = predicted[j, i, :, :]
                plt.pcolor(abs(predicted_sub))
                if(j == 0 and i == 0):
                    plt.colorbar()
                plt.imshow(predicted_sub)
                plt.savefig('./plot/img' + str(fold) + str(j) + str(i) + '.png')

    def plot_MAE(prediction, data_next, test, fold):
        # mae
        MAE = np.zeros((9), dtype=np.double)
        for i in range(9):
            MAE_i = abs(prediction[i, :, :] - data_next[test])
            MAE[i] = np.mean(MAE_i)

        plt.clf()
        k = ['k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7', 'k=8', 'k=9', 'k=10']
        sns.set(style="whitegrid")

        df = pd.DataFrame(dict(x=k, y=MAE))
        # total = sns.load_dataset('tips')
        ax = sns.barplot(x="x", y="y", data=df)
        min = MAE.min() - 0.01
        max = MAE.max() + 0.01
        ax.set(ylim=(min, max))
        plt.savefig('./plot/mae' + str(fold) + '.png')

    ######################################################################################################################################

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()

            nn = Sequential(Linear(1, 1225), ReLU())
            self.conv1 = NNConv(35, 35, nn, aggr='mean', root_weight=True, bias=True)
            self.conv11 = BatchNorm(35, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

            nn = Sequential(Linear(1, 35), ReLU())
            self.conv2 = NNConv(35, 1, nn, aggr='mean', root_weight=True, bias=True)
            self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

            nn = Sequential(Linear(1, 35), ReLU())
            self.conv3 = NNConv(1, 35, nn, aggr='mean', root_weight=True, bias=True)
            self.conv33 = BatchNorm(35, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)



        def forward(self, data):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

            x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
            x1 = F.dropout(x1, training=self.training)

            x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
            x2 = F.dropout(x2, training=self.training)

            embedded = x2.detach().cpu().clone().numpy()

            return embedded

    def embed(Casted_source):
        embedded_data = np.zeros((1, 35), dtype=float)
        i = 0
        for data_A in Casted_source:  ## take a subject from source and target data
            embedded = generator(data_A)  # 35 x35

            if i == 0:
                embedded = np.transpose(embedded)
                embedded_data = embedded
            else:
                embedded = np.transpose(embedded)
                embedded_data = np.append(embedded_data, embedded, axis=0)
            i = i + 1
        return embedded_data

    def test_gGAN(data_next, embedded_train_data, embedded_test_data, embedded_CBT):
        def x_to_x(x_train, x_test, nbr_of_trn, nbr_of_tst):
            result = np.empty((nbr_of_tst, nbr_of_trn), dtype=float)
            for i in range(nbr_of_tst):
                x_t = np.transpose(x_test[i])
                for j in range(nbr_of_trn):
                    result[i, j] = np.matmul(x_train[j], x_t)
            return result

        def check(neighbors, i, j):
            for val in neighbors[i, :]:
               if val == j:
                    return 1
            return 0

        def k_neighbors(x_to_x, k_num, nbr_of_trn, nbr_of_tst):
            neighbors = np.zeros((nbr_of_tst, k_num), dtype=int)
            used = np.zeros((nbr_of_tst, nbr_of_trn), dtype=int)
            current = 0
            for i in range(nbr_of_tst):
                for k in range(k_num):
                    for j in range(nbr_of_trn):
                        if abs(x_to_x[i, j]) > current:
                            if check(neighbors, i, j) == 0:
                                neighbors[i, k] = j
                                current = abs(x_to_x[i, neighbors[i, k]])
                    current = 0

            return neighbors

        def subtract_cbt(x, cbt, length):
            for i in range(length):
                x[i] = abs(x[i] - cbt[0])

            return x

        def predict_samples(k_neighbors, t1, nbr_of_tst):
            average = np.zeros((nbr_of_tst, 595), dtype=float)
            for i in range(nbr_of_tst):
                for j in range(len(k_neighbors[0])):
                    average[i] = average[i] + t1[k_neighbors[i,j],:]

                average[i] = average[i] / len(k_neighbors[0])

            return average

        residual_of_tr_embeddings = subtract_cbt(embedded_train_data, embedded_CBT, len(embedded_train_data))
        residual_of_ts_embeddings = subtract_cbt(embedded_test_data, embedded_CBT, len(embedded_test_data))

        dot_of_residuals = x_to_x(residual_of_tr_embeddings, residual_of_ts_embeddings, len(train), len(test))
        for k in range(2, 11):
            k_neighbors_ = k_neighbors(dot_of_residuals, k, len(train), len(test))

            if k == 2:
                prediction = predict_samples(k_neighbors_, data_next, len(embedded_test_data))
                prediction = np.reshape(prediction, (1, len(embedded_test_data), nbr_of_feat))
            else:
                new_predict = predict_samples(k_neighbors_, data_next, len(embedded_test_data))
                new_predict = np.reshape(new_predict, (1, len(embedded_test_data), nbr_of_feat))
                prediction = np.append(prediction, new_predict, axis=0)

        return prediction

    nbr_of_sub = int(input('Please select the number of subjects: '))
    if nbr_of_sub < 5:
        print("You can not give less than 5 to the number of subjects. ")
        nbr_of_sub = int(input('Please select the number of subjects: '))
    nbr_of_regions = int(input('Please select the number of regions: '))
    nbr_of_epochs = int(input('Please select the number of epochs: '))
    nbr_of_folds = int(input('Please select the number of folds: '))
    hyper_param1 = 100
    nbr_of_feat = int((np.square(nbr_of_regions) - nbr_of_regions) / 2)
    nbr_of_sub_for_cbt = int(nbr_of_sub // 5)  # CBT will be generated by %20 of the number of subjects.
    print(nbr_of_sub_for_cbt)

    data = np.random.normal(0.6, 0.3, (nbr_of_sub, nbr_of_regions, nbr_of_regions))
    independent_data = np.random.normal(0.6, 0.3, (nbr_of_sub_for_cbt, nbr_of_regions, nbr_of_regions))
    data_next = np.random.normal(0.4, 0.3, (nbr_of_sub, nbr_of_regions, nbr_of_regions))
    CBT = netNorm(independent_data, nbr_of_sub_for_cbt, nbr_of_regions)
    gGAN(data, nbr_of_regions, nbr_of_epochs, nbr_of_folds, hyper_param1, CBT)

    # embed train and test subjects
    kfold = KFold(n_splits=nbr_of_folds, shuffle=True, random_state=manualSeed)

    source_data = torch.from_numpy(data)  # convert numpy array to torch tensor
    source_data = source_data.type(torch.FloatTensor)

    target_data = np.reshape(CBT, (1, nbr_of_regions, nbr_of_regions, 1))
    target_data = torch.from_numpy(target_data)  # convert numpy array to torch tensor
    target_data = target_data.type(torch.FloatTensor)

    i = 1
    for train, test in kfold.split(source_data):
        adversarial_loss = torch.nn.BCELoss()
        l1_loss = torch.nn.L1Loss()
        trained_model_gen = torch.load('./weight_' + str(i) + 'generator_.model')
        generator = Generator()
        generator.load_state_dict(trained_model_gen)

        train_data = source_data[train]
        test_data = source_data[test]

        generator.to(device)
        adversarial_loss.to(device)
        l1_loss.to(device)

        X_train_casted_source = [d.to(device) for d in cast_data(train_data, 0)]
        X_test_casted_source = [d.to(device) for d in cast_data(test_data, 0)]
        data_B = [d.to(device) for d in cast_data(target_data, 0)]

        embedded_train_data = embed(X_train_casted_source)
        embedded_test_data = embed(X_test_casted_source)
        embedded_CBT = embed(data_B)

        if i == 1:
            data_next = linear_features(data_next)
        predicted_flat = test_gGAN(data_next, embedded_train_data, embedded_test_data, embedded_CBT)

        plot_MAE(predicted_flat, data_next, test, i)
        i = i + 1

        predicted = make_sym_matrix(nbr_of_regions, predicted_flat)
        plot_predictions(predicted, i - 1)

demo()

