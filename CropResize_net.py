# -*- coding: utf-8 -*-
# @Time    : 2023/7/23 15:54
# @Author  : Cool_B
# @Email    : 467387544@qq.com
# @File    : CropResizeBranch.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

class CropResizeBranch(nn.Module):
    def __init__(self):
        super(CropResizeBranch, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64*56*56, 1024)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        features = self.fc(x.view(x.size(0), -1))
        return features


class EraseComplementaryBranch(nn.Module):
    def __init__(self):
        super(EraseComplementaryBranch, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64*56*56, 1024)  # Assuming output size of the Conv layer is 56x56

    def forward(self, x):

        x = F.relu(self.conv1(x))
        features = self.fc(x.view(x.size(0), -1))
        return features

class CustomTransform:
    def __init__(self, crop_size=224, erase_size=56):
        self.crop_size = crop_size
        self.erase_size = erase_size

    def __call__(self, img):

        crop_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.crop_size),
            transforms.Resize((self.crop_size, self.crop_size)),
        ])
        cropped_img = crop_transform(img)


        erase_transform = transforms.Compose([
            transforms.Resize((self.erase_size, self.erase_size)),
        ])
        erased_img = erase_transform(img)

        return cropped_img, erased_img


data_dir = 'path_to_cub200_dataset'
train_transform = transforms.Compose([
    CustomTransform(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


num_classes = len(train_dataset.classes)
model = InceptionV3WithBranches(num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Crop and Resize sensitive area
        cropped_inputs = inputs[:, :, :56, :56]
        cropped_inputs = F.interpolate(cropped_inputs, size=(224, 224), mode='bilinear')

        optimizer.zero_grad()
        outputs = model(cropped_inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} Accuracy: {100. * correct / total:.2f}%")

    print(f"Epoch [{epoch}/{num_epochs}] Training Loss: {train_loss / len(train_loader):.4f} Training Accuracy: {100. * correct / total:.2f}%")


model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)


        cropped_inputs = inputs[:, :, :56, :56]
        cropped_inputs = F.interpolate(cropped_inputs, size=(224, 224), mode='bilinear')

        outputs = model(cropped_inputs)

        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f"Testing Loss: {test_loss / len(test_loader):.4f} Testing Accuracy: {100. * correct / total:.2f}%")
