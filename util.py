from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torchvision

from dataset import ARIL_dataset, SignFi_dataset
from model import ARIL_ResNet18, ARIL_ResNet50, SignFi_ResNet50
from models.model_CBAM import ARIL_ResNet18_CBAM
from models.model_CBAM2 import ARIL_ResNet18_CBAM2
from models.model_CBAM2_mish import ARIL_ResNet18_CBAM2_mish
from models.model_CBAM2_parallel import ARIL_ResNet18_CBAM2_parallel


def load_data_n_model(dataset_name, model_name, root, test):
    channels = {'ARIL': 52, 'SignFi': 90}
    classes = {'ARIL': 6, 'SignFi': 276}
    if dataset_name == 'ARIL':
        num_classes = classes['ARIL']
        num_channels = channels['ARIL']
        print('using dataset: ARIL')
        train_data, train_label, test_data, test_label = ARIL_dataset(root)

        train_set = TensorDataset(train_data, train_label)
        test_set = TensorDataset(test_data, test_label)

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                                  drop_last=True)  # drop_last=True
        test_loader = DataLoader(test_set, batch_size=64, shuffle=True,
                                 drop_last=True)  # drop_last=True

        if model_name == 'ResNet18':
            print("using model: ResNet18")
            model = ARIL_ResNet18(num_classes, num_channels)
            train_epoch = 100  # 70

        elif model_name == "ResNet18_CBAM":
            print("using model: ResNet18_CBAM")
            model = ARIL_ResNet18_CBAM(num_classes, num_channels)
            train_epoch = 100  # 70

        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = ARIL_ResNet50(num_classes, num_channels)
            train_epoch = 100  # 70

        elif model_name == 'ResNet18_CBAM2':
            print("using model: ResNet18_CBAM2")
            model = ARIL_ResNet18_CBAM2(num_classes, num_channels)
            train_epoch = 200  # 70

        elif model_name == 'ResNet18_CBAM2_mish':
            print("using model: ResNet18_CBAM2_mish")
            model = ARIL_ResNet18_CBAM2_mish(num_classes, num_channels)
            train_epoch = 100  # 70

        elif model_name == 'ARIL_ResNet18_CBAM2_parallel':
            print("using model: ARIL_ResNet18_CBAM2_parallel")
            model = ARIL_ResNet18_CBAM2_parallel(num_classes, num_channels)
            train_epoch = 200  # 70

    elif dataset_name == 'SignFi':
        num_classes = classes['SignFi']
        num_channels = channels['SignFi']
        print('using dataset: SignFi')
        train_data, train_label, test_data, test_label = SignFi_dataset(root)

        train_set = TensorDataset(train_data, train_label)
        test_set = TensorDataset(test_data, test_label)

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                                  drop_last=True)  # drop_last=True
        test_loader = DataLoader(test_set, batch_size=64, shuffle=True,
                                 drop_last=True)  # drop_last=True

        if model_name == 'ResNet18':
            print("using model: ResNet18")
            model = ARIL_ResNet18(num_classes, num_channels)
            train_epoch = 100  # 70

        elif model_name == "ResNet18_CBAM":
            print("using model: ResNet18_CBAM")
            model = ARIL_ResNet18_CBAM(num_classes, num_channels)
            train_epoch = 100  # 70

        elif model_name == "ResNet50":
            print("using model: ResNet50")
            model = SignFi_ResNet50(num_classes, num_channels)
            train_epoch = 100  # 70

    if test:
        train_epoch = 2

    return train_loader, test_loader, model, train_epoch


