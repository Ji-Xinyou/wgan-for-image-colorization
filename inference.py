import csv
import time
from turtle import color

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from data import get_miniImagenet_dataset


def get_model(epoch):
    print("========== Loading model....")
    model = torch.load(f'./G_conv_epoch{epoch}.pth')
    print("========== Model loaded!")
    return model

def get_test_loader():
    print("========== Loading dataset")
    test_loader = get_miniImagenet_dataset(batchsize=64, shape=(128, 128), phase='test')
    print("========== Dataset loaded!")
    return test_loader

def demo_inference():
    model = get_model(19).cuda()
    model.requires_grad = False
    
    path = None
    with open("./data/mini-imagenet/test.csv", mode='r') as f:
        r = np.random.randint(1, 12002)
        reader = csv.reader(f)
        filename = ""
        for i, row in enumerate(reader):
            if i == r:
                filename = row[0]
                break
        path = f"./data/mini-imagenet/images/{filename}"

    print(path)
    colored = cv2.imread(path)
    gray = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    gray = cv2.resize(gray, [128, 128])
    colored = cv2.resize(colored, [128, 128])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=[128, 128]),
        transforms.ToTensor(),
    ])
    tensor_gray = transform(gray)
    tensor_gray = torch.cat([tensor_gray, tensor_gray, tensor_gray], dim=0)
    tensor_gray = tensor_gray.unsqueeze(0).cuda()

    toc1 = time.perf_counter()
    gen = model(tensor_gray).detach().cpu().squeeze(0)
    toc2 = time.perf_counter()

    print("===============================")
    print("|")
    print(f"| used {toc2-toc1} second to generate a image")
    print("|")
    print("===============================")
    
    gen = np.transpose(gen, (1, 2, 0))
    print("done generating")

    plt.subplot(1, 3, 1)
    plt.title("gray scale image")
    plt.imshow(gray, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("ground truth")
    plt.imshow(colored)

    plt.subplot(1, 3, 3)
    plt.title("Our Model")
    plt.imshow(gen, vmin=0, vmax=1)

    plt.show()

if __name__ == "__main__":
    demo_inference()
