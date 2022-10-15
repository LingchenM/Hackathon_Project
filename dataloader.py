import torch
import random
import numpy as np
from PIL import Image

def trainLoader(transform):
    trainData = []
    for i in range(0,10):
        for j in range(1,3):
            for k in range(0, 200):
                img = Image.open('D:\Desktop\Hack Project\hackathon\credit_card\credit_card_original\\train\\' + str(i) + '\\' + '_' + str(j) + '_' + str(k) + '.jpg')
                img = transform(img)
                label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                label[i] = 1
                label = torch.Tensor(label)
                trainData.append([img, label])
    random.shuffle(trainData)
    return trainData

def testLoader(transform):
    testData = []
    for i in range(0, 10):
        for j in range(1, 3):
            for k in range(0, 200):
                img = Image.open('D:\Desktop\Hack Project\hackathon\credit_card\credit_card_original\\test\\' + str(i) + '\\' + '_' + str(j) + '_' + str(k) + '.jpg')
                img = transform(img)
                label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                label[i] = 1
                label = torch.Tensor(label)
                testData.append([img, label])
    random.shuffle(testData)
    return testData