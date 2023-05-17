import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from attention import CBAM


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ARIL_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_channels=52, num_classes=7):
        super(ARIL_ResNet, self).__init__()
        self.reshape = nn.Sequential(
            nn.Conv2d(1, 3, 7, stride=(3, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, kernel_size=(10, 11), stride=1),
            nn.ReLU()
        )
        self.in_channels = 64
        self.num_classes = num_classes
        self.CBAM = CBAM(num_channels)  # 通道注意力机制
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        self.CBAM_1 = CBAM(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(512 * ResBlock.expansion, 256)

        self.fc_2 = nn.Linear(256, num_classes)

        self.fc_3 = nn.Linear(512 * ResBlock.expansion, num_classes)

        self.m = nn.Mish()

    def forward(self, x):
        x = self.CBAM(x)
        # x = self.reshape(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.CBAM_1(x)
        x = self.m(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        if self.num_classes < 256:

            x = self.fc_1(x)
            # x = nn.Dropout(0.5)(x)
            x = self.fc_2(x)
        else:
            x = nn.Dropout(0.5)(x)
            x = self.fc_3(x)
        return x


    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ARIL_ResNet18_CBAM2_mish(num_classes, num_channels):
    return ARIL_ResNet(Block, [2, 2, 2, 2], num_classes=num_classes, num_channels=num_channels)


if __name__ == '__main__':
    input = torch.ones((4, 52, 512, 512))
    model = ARIL_ResNet18_CBAM2_mish(num_classes=7, num_channels=52)
    output = model(input)
    print(output.shape)
