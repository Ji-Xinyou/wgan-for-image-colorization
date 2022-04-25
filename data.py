from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv

class miniImagenet(Dataset):
    def __init__(self, phase='train', transforms=None):
        super().__init__()
        self.transforms = transforms
        self.paths = []

        assert(phase == 'train' or phase == 'test' or phase == 'val' or phase == None)

        path = './data/mini-imagenet/' + phase + '.csv'
        with open(path, mode='r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i != 0:
                    self.paths.append('./data/mini-imagenet/images/' + row[0])
            
    
    def __getitem__(self, index):
        path = self.paths[index] # path of colored image

        y_tr = cv2.imread(path)
        x_tr = cv2.cvtColor(y_tr, cv2.COLOR_BGR2GRAY)
        y_tr = cv2.cvtColor(y_tr, cv2.COLOR_BGR2RGB)

        if self.transforms:
            x_tr, y_tr = self.transforms(x_tr), self.transforms(y_tr)
        
        return (x_tr, y_tr)
    
    def __len__(self):
        return len(self.paths)

def get_miniImagenet_dataset(batchsize=64, shape=(224, 224), phase='train'):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=[*shape]),
        transforms.ToTensor(),
    ])

    dataset = miniImagenet(phase=phase, transforms=transform)
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batchsize)

    return dataloader

def get_miniImagenet_train_val_dataset(batchsize=32, shape=(128, 128)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(size=[*shape]),
        transforms.ToTensor(),
    ])
    return

def demo():
    train_loader = get_miniImagenet_dataset(batchsize=64, phase='train')
    val_loader = get_miniImagenet_dataset(batchsize=64, phase='train')
    test_loader = get_miniImagenet_dataset(batchsize=64, phase='train')
    for i, (x_tr, y_tr) in enumerate(train_loader):
        # x_tr (C, H, W)
        if i == 0:
            plt.imshow(np.transpose(x_tr[0].numpy()), cmap="gray")
            plt.show()
            plt.imshow(np.transpose(y_tr[0].numpy()))
            plt.show()
            break
    for i, (x_tr, y_tr) in enumerate(val_loader):
        # x_tr (C, H, W)
        if i == 0:
            plt.imshow(np.transpose(x_tr[0].numpy()), cmap="gray")
            plt.show()
            plt.imshow(np.transpose(y_tr[0].numpy()))
            plt.show()
            break
    for i, (x_tr, y_tr) in enumerate(test_loader):
        # x_tr (C, H, W)
        if i == 0:
            plt.imshow(np.transpose(x_tr[0].numpy()), cmap="gray")
            plt.show()
            plt.imshow(np.transpose(y_tr[0].numpy()))
            plt.show()
            break

if __name__ == "__main__":
    demo()