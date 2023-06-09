import numpy as np
import glob
import scipy.io as sio
import torch
import mat73
from torch.utils.data import Dataset, DataLoader


def ARIL_dataset(root):
    train_data_path = root + 'ARIL/' + 'ARIL_train.mat'
    test_data_path = root + 'ARIL/' + 'ARIL_test.mat'

    # load train data
    tr_data = mat73.loadmat(train_data_path)
    train_data = tr_data['ARIL_train_data']

    train_label = tr_data['ARIL_train_label']
    train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.LongTensor)

    # load eval data
    te_data = mat73.loadmat(test_data_path)
    test_data = te_data['ARIL_test_data']

    test_label = te_data['ARIL_test_label']

    test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)

    return train_data, train_label, test_data, test_label


def SignFi_dataset(root):
    # train data
    origin_train_data_path = root + 'SignFi/' + 'SignFi_train_data'
    train_data_list = []
    print("load data: ")
    for i in range(1, 1):
        path = origin_train_data_path + str(i) + '.mat'
        print(path)
        tr_data = mat73.loadmat(path)
        train_data = tr_data['data']
        train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
        train_data_list.append(train_data)

    train_data = torch.cat(train_data_list, dim=0)
    print("train_data shape: ")
    print(train_data.shape)

    train_label_path = root + 'SignFi/' + 'SignFi_train_label.mat'
    tr_label = sio.loadmat(train_label_path)
    train_label = tr_label['label_lab']
    assert train_label.shape == (5520, 1)

    train_label = train_label.astype('uint8')
    # train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.LongTensor)
    train_label -= 1

    test_data_path = root + 'SignFi/' + 'SignFi_test.mat'
    # load eval data
    te_data = mat73.loadmat(test_data_path)
    test_data = te_data['SignFi_test_data']

    test_label = te_data['SignFi_test_label']

    test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)

    return train_data, train_label, test_data, test_label
