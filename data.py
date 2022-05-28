import matplotlib.pyplot as plt
import random
import numpy as np
import glob
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class miniImagenet(Dataset):
    def __init__(self, args, phase='train', size=8000, transforms=None):
        super().__init__()
        self.transform = transforms
        self.paths = []

        path = f"{args.dataroot}/{phase}"
        self.paths = glob.glob(f"{path}/*.JPEG")
        random.shuffle(self.paths)
        self.paths = self.paths[:min(len(self.paths), size)]

        print(f"Dataset size: {len(self.paths)} images")

    def __getitem__(self, index):
        path = self.paths[index]  # path of colored image

        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        image = np.array(image)
        lab = rgb2lab(image).astype("float32")
        lab = transforms.ToTensor()(lab)

        l = lab[[0], ...] / 100 - 1  # [-1, 1]
        ab = lab[[1, 2], ...] / 110  # [-1, 1]

        return (l, ab)

    def __len__(self):
        return len(self.paths)

def get_miniImagenet_dataset(args, batchsize=64, size=8000, shape=(224, 224), phase='train'):
    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=[*shape]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=[*shape]),
        ])

    dataset = miniImagenet(args=args, phase=phase, size=size, transforms=transform)
    shuffle = False if phase == 'test' else True
    dataloader = DataLoader(
        dataset=dataset, shuffle=shuffle, batch_size=batchsize)

    return dataloader

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
