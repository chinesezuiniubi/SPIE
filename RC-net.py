# -*- coding: utf-8 -*-
# @Time    : 2023/7/23 15:46
# @Author  : Cool_B
# @Email    : 467387544@qq.com
# @File    : RC-net.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class InceptionV3WithBranches(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3WithBranches, self).__init__()
        self.inception_v3 = models.inception_v3(pretrained=True)

        self.inception_v3.fc = nn.Identity()
        self.inception_v3.AuxLogits = nn.Identity()
        self.branch1 = CropResizeBranch()
        self.branch2 = EraseComplementaryBranch()

        self.classifier = nn.Linear(2048 + 1024 + 1024, num_classes)
    def forward(self, x):

        inception_features = self.inception_v3(x)


        branch1_features = self.branch1(x)
        branch2_features = self.branch2(x)

        combined_features = torch.cat((inception_features, branch1_features, branch2_features), dim=1)


        output = self.classifier(combined_features)
        return output


class CropResizeBranch(nn.Module):
    def __init__(self):
        super(CropResizeBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # ...
        self.fc = nn.Linear(output_size, 1024)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        features = self.fc(x.view(x.size(0), -1))
        return features


class EraseComplementaryBranch(nn.Module):
    def __init__(self):
        super(EraseComplementaryBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # ...
        self.fc = nn.Linear(output_size, 1024)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        features = self.fc(x.view(x.size(0), -1))
        return features


model = InceptionV3WithBranches(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
