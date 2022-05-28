import argparse
import datetime
import os
import torch
import torch.nn as nn

from tqdm import tqdm
from data import get_miniImagenet_dataset
from model import MainModel, NLayerDiscriminator, Sobel
from unet import ResNetUNet
from utils import visualize

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10,
                    help="number of epochs of training")
parser.add_argument("--pretrain_epochs", type=int, default=50, help="")
parser.add_argument("--batch_size", type=int, default=16,
                    help="size of the batches")
parser.add_argument("--dataset", type=str, default='imagenet')
parser.add_argument("--weight_decay", type=float, default=1e-4, help=""),
parser.add_argument("--lr_G", type=float, default=1e-4, help="learning rate")
parser.add_argument("--lr_D", type=float, default=1e-4, help="learning rate")
parser.add_argument("--n_critic", type=int, default=5,
                    help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=1,
                    help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int,
                    default=300, help="interval between image samples")
parser.add_argument("--saving_interval", type=int, default=50,
                    help="interval between model saving")
parser.add_argument("--dataroot", type=str, default="../../data/imagenet")
parser.add_argument("--c", type=float, default=1000, help="constant of L1 norm")
parser.add_argument("--gp", type=float, default=10, help="constant of gp")
parser.add_argument("--ckpt_path", type=str, default='./checkpoint', help="")
parser.add_argument("--name", type=str, default='experiment_name', help="")
parser.add_argument("--verbose_interval", type=float, default=20, help="")
parser.add_argument("--load_pretrained", type=bool, default=False, help="")
parser.add_argument("--pretrained_path", type=str,
                    default='./checkpoint/experiment_name/weights/pretrained_G.pth', help="")
parser.add_argument("--continue_train", type=bool, default=False)
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--model_path", type=str,
                    default='./checkpoint/experiment_name/weights/checkpoint_last.pth')
parser.add_argument("--wgan", action='store_true')
parser.add_argument("--edge", action='store_true')
args = parser.parse_args()

print(args)

tensorType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def train(G, D, train_loader, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, D = G.to(device), D.to(device)
    # first train G
    print("Training G...........")
    pre_criterion = nn.SmoothL1Loss()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr_G, weight_decay=args.weight_decay)
    edge_detector = Sobel().to(device)
    G.train()
    for epoch in tqdm(range(args.pretrain_epochs)):
        for i, (l, ab) in enumerate(train_loader):
            input, ab = l.to(device), ab.to(device)
            l = l.to(device)
            if args.edge:
                edge = edge_detector(l)
                input = torch.cat([l, edge], dim=1)
            gen_ab = G(input)
            optimizer_G.zero_grad()
            loss = pre_criterion(gen_ab, ab)
            loss.backward()
            optimizer_G.step()

            if i % args.verbose_interval == 0:
                print(f"|Train G| Epoch: {epoch}/{args.pretrain_epochs} Batch: {i}/{len(train_loader)} Loss: {loss.item()}")
    
    # finally GAN
    batches_done = 0
    with open(f'{args.ckpt_path}/{args.name}/train_log.txt', mode='w') as f:
        pass

    model = MainModel(G, D, args.lr_G, args.lr_D, lambda_L1=args.c,
                      lambda_gp=args.gp, clip_value=args.clip_value,
                      weight_decay=args.weight_decay, clip=args.wgan,
                      use_edge=args.edge).cuda()

    for epoch in tqdm(range(args.n_epochs)):
        with open(f'{args.ckpt_path}/{args.name}/train_log.txt', mode='a') as f:
            f.write(str(datetime.datetime.now()) + '\n')
        (val_l, val_ab) = next(iter(val_loader))
        for i, (l, ab) in enumerate(train_loader):
            model.setup_input(l, ab)
            model.optimize()

            losses = model.loss_item()

            if i % args.verbose_interval == 0:
                with open(f'{args.ckpt_path}/{args.name}/train_log.txt', mode='a') as f:
                    f.write(
                        f"[Epoch {epoch}/{args.n_epochs}] [Batch {batches_done % len(train_loader)}/{len(train_loader)}] [D: {losses['D']}] [G_GAN: {losses['G_GAN']}] [G_L1: {losses['G_L1']}]" + '\n')

            if batches_done % args.sample_interval == 0:
                visualize(args, epoch, model, val_l, val_ab, True)

            batches_done += 1

        if epoch % args.saving_interval == 0:
            print(f"Saving model on epoch{epoch + 1}")
            torch.save(
                model, f"{args.ckpt_path}/{args.name}/weights/checkpoint_epoch{epoch+1}.pth")
            print(f"model saved on epoch{epoch + 1}")

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_shape = (args.size, args.size)

    if args.dataset == 'imagenet':
        train_loader = get_miniImagenet_dataset(
            args=args, batchsize=args.batch_size, shape=default_shape, size=10000, phase='train')
        val_loader = get_miniImagenet_dataset(
            args=args, batchsize=args.batch_size, shape=default_shape, size=10000, phase='val')
        test_loader = get_miniImagenet_dataset(
            args=args, batchsize=args.batch_size, shape=default_shape, size=10000, phase='test')
    else:
        raise ValueError("Wrong dataset name")

    os.makedirs(f'{args.ckpt_path}/{args.name}/images', exist_ok=True)
    os.makedirs(f'{args.ckpt_path}/{args.name}/weights', exist_ok=True)

    with open(f"{args.ckpt_path}/{args.name}/train_args.txt", mode='w') as f:
        for arg in vars(args):
            f.write(format(arg, '<20'))
            f.write(format(str(getattr(args, arg)), '<') + '\n')

    inchan = 3 if args.edge else 1
    G = ResNetUNet(in_class=inchan, n_class=2)
    print(inchan)
    D = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d)

    model = train(G, D, train_loader, val_loader, args)

    torch.save(model, f'{args.ckpt_path}/{args.name}/weights/checkpoint_last.pth')