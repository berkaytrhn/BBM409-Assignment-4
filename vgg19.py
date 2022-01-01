import cv2
import natsort as natsort
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL.Image import Image
from torch import optim
from torch.utils import data
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import os
from torch.utils.data import DataLoader, random_split
import math
from time import time

"""
class VGG19(nn.Module):

    def __init__(self, finetune_all=True, num_classes=10):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        vgg19 = models.vgg19(pretrained=True)
        removed = list(vgg19.classifier.children())[:-1]
        vgg19.classifier = nn.Sequential(*removed)
        vgg19.classifier = nn.Sequential(*list(vgg19.classifier.children()), nn.Linear(in_features=4096, out_features=num_classes, bias=True))
        super(VGG19, self).__init__()
        self.model = vgg19
        print(self.model)

        if not finetune_all:
            # Freeze those weights
            for p in self.model.features.parameters():
                p.requires_grad = False

    def forward(self, x):
            return self.vgg19(x)
"""

def Test(epoch, model, epochBased, testData, criterion, device):
    totalLoss = 0
    start = time()

    accuracy = []

    with torch.no_grad():  # disable calculations of gradients for all pytorch operations inside the block
        for i, batch in enumerate(testData):
            minput = batch[0].to(device)
            target = batch[1].to(device)

            # output by our model
            moutput = model(minput)

            # computing the cross entropy loss
            loss = criterion(moutput, target)
            totalLoss += loss.item()


            argmax = moutput.argmax(dim=1)
            # Find the accuracy of the batch by comparing it with actual targets
            accuracy.append((target == argmax).sum().item() / target.shape[0])

    if epochBased:
        print('Epoch: [{}], Test Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(epoch, totalLoss / len(testData), sum(accuracy) / len(accuracy), time() - start))
    # Returning Average Testing Loss and Accuracy
    return totalLoss / len(testData), sum(accuracy) / len(accuracy)


def Train(epoch, model, print_every, trainData, criterion, device, optimizer):
    # method for training model
    totalLoss = 0
    start = time()

    accuracy = []

    for i, batch in enumerate(trainData, 1):

        minput = batch[0].to(device)
        target = batch[1].to(device)

        # output of model
        moutput = model(minput)

        # computing the cross entropy loss
        loss = criterion(moutput, target)
        totalLoss += loss.item()

        optimizer.zero_grad()

        # Back propogation
        loss.backward()

        # updating model parameters
        optimizer.step()

        argmax = moutput.argmax(dim=1)
        # calculating accuracy by comparing to target
        accuracy.append((target == argmax).sum().item() / target.shape[0])

        if i % print_every == 0:
            print('Epoch: {}, Train Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(epoch, loss.item(),
                                                                                             sum(accuracy) / len(
                                                                                                 accuracy),
                                                                                             time() - start))
    # Returning Average Training Loss and Accuracy
    return totalLoss / len(trainData), sum(accuracy) / len(accuracy)


def epochLoop(epochNumber, model, trainData, testData, criterion, device, optimizer, batchSize, learningRate):
    # main loop
    trainLosses = []
    testLosses = []
    trainAccuracies = []
    testAccuracies = []

    for epoch in range(1, epochNumber + 1):
        trainLoss, trainAccuracy = Train(epoch, model, 1, trainData, criterion, device, optimizer)
        testLoss, testAccuracy = Test(epoch, model, True, testData, criterion, device)

        trainLosses.append(trainLoss)
        testLosses.append(testLoss)
        trainAccuracies.append(trainAccuracy)
        testAccuracies.append(testAccuracy)

        print('\n')

        if epoch % 30 == 0:
            # SAVING MODEL TO USE IT LATER
            torch.save(model, f"./model_epoch-30_lr-{learningRate}_batch-{batchSize}_dropout-0.45.pth")
    return trainLosses, testLosses, trainAccuracies, testAccuracies

def defineTransforms(IMG_SIZE):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=40),

            transforms.Resize((IMG_SIZE,IMG_SIZE)),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )
    return transform

def loadData(testPercent, batchSize, dataset):
    # train test set separation
    trainNumber = math.floor((100 - testPercent) * len(dataset) / 100)
    testNumber = math.ceil(testPercent * len(dataset) / 100)
    trainSet, testSet = random_split(dataset, (trainNumber, testNumber))#, generator=torch.Generator().manual_seed(42))
    # initializing data loaders
    trainData = DataLoader(
        dataset=trainSet,
        batch_size=batchSize,
        shuffle=True,
        num_workers=1
    )

    testData = DataLoader(
        dataset=testSet,
        batch_size=batchSize,
        shuffle=True,
        num_workers=1
    )
    return trainData, testData

def initModel(device, learningRate, finetune_all=None, num_classes=10):
    # initializing model

    vgg19 = models.vgg19(pretrained=True)
    removed = list(vgg19.classifier.children())[:-1]
    vgg19.classifier = nn.Sequential(*removed)
    vgg19.classifier = nn.Sequential(*list(vgg19.classifier.children()), nn.Linear(in_features=4096, out_features=num_classes, bias=True))

    if not finetune_all:
        # Freeze those weights
        for p in vgg19.features.parameters():
            p.requires_grad = False

    vgg19 = vgg19.to(device)

    # loss and optimization function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg19.parameters(), lr=learningRate)

    return vgg19, criterion, optimizer


def set_device():
    # set device
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using", device)
    return device

def main():
    BATCH_SIZE=32
    IMAGE_SIZE=224

    EPOCHS = 50
    LEARNING_RATE = 0.001

    transforms = defineTransforms(IMAGE_SIZE)


    dataset = ImageFolder("dataset", transforms)

    train_data, test_data = loadData(30, BATCH_SIZE, dataset)



    device = set_device()

    model, criterion, optimizer = initModel(device, LEARNING_RATE)

    trainLosses, testLosses, trainAccuracies, testAccuracies = epochLoop(EPOCHS, model, train_data, test_data, criterion, device, optimizer, BATCH_SIZE, LEARNING_RATE)






if __name__ == "__main__":
    main()