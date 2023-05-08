from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torchvision

from dataset import ARIL_dataset
from model import ARIL_ResNet18
from models.model_CBAM import ARIL_ResNet18_CBAM


def load_data_n_model(dataset_name, model_name, root):
    classes = {'ARIL': 6, 'SignFi': 276}
    if dataset_name == 'ARIL':
        num_classes = classes['ARIL']
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
            model = ARIL_ResNet18(num_classes)
            train_epoch = 200  # 70

        elif model_name == "ResNet18_CBAM":
            print("using model: ResNet18_CBAM")
            model = ARIL_ResNet18_CBAM(num_classes)
            train_epoch = 100  # 70

    return train_loader, test_loader, model, train_epoch


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # out = self.spatial_attention(out) * out
        return out
