import glob
import numpy as np
import time
import torch
import argparse
import glob
import torchvision.transforms as transforms
from unet import ResNetUNet

from PIL import Image
from utils import lab_to_rgb

parser = argparse.ArgumentParser()
parser.add_argument("--weight_path", required=True,
                    type=str, default='./checkpoint')
parser.add_argument("--gray_dir", required=True, type=str,
                    default="../../data/imagenet")
parser.add_argument("--out_dir", required=True,
                    type=str, default="./out_results")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model():
    model = torch.load(args.weight_path)
    return model

def demo_inference(args):
    model = get_model().to(device)
    model.requires_grad = False

    paths = glob.glob(f"{args.gray_dir}/*")
    for i, path in enumerate(paths):
        img = Image.open(path).convert("L")
        img = np.array(img)
        img = img / 255 - 1
        img = img.astype("float32")
        l = transforms.ToTensor()(img)
        l = transforms.Resize((256, 256))(l)
        l = torch.unsqueeze(l, dim=0).to(device)
        ab = torch.zeros((1, 2, 256, 256)).to(device)

        model.net_G.eval()
        with torch.no_grad():
            tic1 = time.time()
            model.setup_input(l, ab)
            model.forward()
            tic2 = time.time()
        model.net_G.train()

        fake_color = model.fake_color.detach()
        L = model.L

        fake_img = lab_to_rgb(L, fake_color)[0] * 255
        fake = Image.fromarray(np.uint8(fake_img))

        save_path = f'{args.out_dir}/{i}.png'
        fake.save(save_path)

        print(f"Colorized and Saved {i+1}/{len(paths)} images, took {tic2-tic1} sec")

if __name__ == "__main__":
    demo_inference(args)
