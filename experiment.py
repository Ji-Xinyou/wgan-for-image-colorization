import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
import cv2

from skimage.metrics import structural_similarity
from PIL import Image
from skimage.color import rgb2lab
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from utils import lab_to_rgb

parser = argparse.ArgumentParser()
parser.add_argument("--all_ckpt", type=str, required=True)
parser.add_argument("--noWgan_ckpt", type=str)
parser.add_argument("--noSobel_ckpt", type=str)
parser.add_argument("--testpath", type=str, required=True)
parser.add_argument("--outpath", type=str, required=True)
parser.add_argument("--experiment", type=str, required=True)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GeneralDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Resize(size=(256, 256))
        self.paths = []
        self.paths = glob.glob(f"{args.testpath}/*")
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

def turn_to_gray():
    paths = glob.glob(f"{args.testpath}/*")
    for i, path in tqdm(enumerate(paths)):
        img = Image.open(path).convert("RGB")
        img = transforms.Resize(size=(256, 256))(img)
        img = np.array(img)
        lab = rgb2lab(img).astype("float32")
        l = lab[..., 0] * 255 / 100 # [0, 255] => -1 before input to model
        l = Image.fromarray(l).convert("L")
        l.save(f'{args.outpath}/{i}.png')

def integrated_test(model):
    paths = glob.glob(f"{args.testpath}/*")
    psnr = 0
    ssim = 0
    for path in tqdm(paths):
        img = Image.open(path).convert("RGB")
        img = transforms.Resize(size=(256, 256))(img)
        img = np.array(img)
        lab = rgb2lab(img).astype("float32")
        lab = transforms.ToTensor()(lab)

        l = lab[[0], ...] / 100 - 1  # [-1, 1]
        l = torch.unsqueeze(l, dim=0)
        ab = lab[[1, 2], ...] / 110  # [-1, 1]
        ab = torch.unsqueeze(ab, dim=0)
        
        model.net_G.eval()
        with torch.no_grad():
            model.setup_input(l, ab)
            model.forward()
        model.net_G.train()

        fake_color = model.fake_color.detach()
        real_color = model.ab
        L = model.L

        fake_img = lab_to_rgb(L, fake_color)[0]
        real_img = lab_to_rgb(L, real_color)[0]

        mse = np.mean((fake_img - real_img) ** 2)
        cur = 10 * np.log10((1 ** 2) / mse)
        psnr += cur

        curssim = structural_similarity(fake_img, real_img, channel_axis=2)
        ssim += curssim

    psnr /= len(paths)
    ssim /= len(paths)
    print(f"Average PSNR: {psnr}")
    print(f"Average SSIM: {ssim}")


def inf_images():
    model = torch.load(args.all_ckpt).to(device) # best model

    dataset = GeneralDataset()
    loader = DataLoader(dataset=dataset, batch_size=1)

    for i, (l, ab) in tqdm(enumerate(loader)):
        model.net_G.eval()
        with torch.no_grad():
            tic1 = time.time()
            model.setup_input(l, ab)
            model.forward()
            tic2 = time.time()
        print(f"Inference takes {tic2 - tic1} seconds")

        model.net_G.train()

        fakeab = model.fake_color.detach().cpu()
        fake_img = lab_to_rgb(l, fakeab)[0]
        l = torch.squeeze(l, dim=0)

        plt.cla()
        plt.subplot(1, 2, 1)
        plt.imshow(l[0].cpu(), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(fake_img)
        plt.savefig(f"{args.outpath}/{i}.png")

def ablation(model_all, model_noedge, model_nowgan):
    models = [model_all, model_noedge, model_nowgan]

    dataset = GeneralDataset()
    loader = DataLoader(dataset=dataset, batch_size=1)

    for i, (l, ab) in tqdm(enumerate(loader)):
        for model in models:
            model.net_G.eval()
            with torch.no_grad():
                tic1 = time.time()
                model.setup_input(l, ab)
                model.forward()
                tic2 = time.time()
            print(f"Inference takes {tic2 - tic1} seconds")
            model.net_G.train()

        fake_images = []
        for model in models:
            fakeab = model.fake_color.detach().cpu()
            fake_img = lab_to_rgb(l, fakeab)[0]
            fake_images.append(fake_img)

        l = torch.squeeze(l, dim=0)
        plt.cla()
        plt.figure(figsize=(5, 20))
        plt.subplot(4, 1, 1)
        plt.imshow(l[0].cpu(), cmap='gray')
        plt.subplot(4, 1, 2)
        plt.imshow(fake_images[0])
        plt.subplot(4, 1, 3)
        plt.imshow(fake_images[1])
        plt.subplot(4, 1, 4)
        plt.imshow(fake_images[2])
        plt.savefig(f"{args.outpath}/{i}.png")

if __name__ == '__main__':
    model1 = torch.load(args.all_ckpt).to(device)
    model2 = torch.load(args.noWgan_ckpt).to(device)
    model3 = torch.load(args.noSobel_ckpt).to(device)
    # carry out your experiments!
    if args.experiment == "ablation":
        ablation(model1, model3, model2)
    elif args.experiment == "integrated":
        for model in [model1, model2, model3]:
            integrated_test(model)