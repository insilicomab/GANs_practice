# ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision.transforms as transforms

import os


'''Googleドライブマウント'''

from google.colab import drive
drive.mount('/content/drive')


'''MNISTデータセットをダウンロード'''

import torchvision.datasets as dset

dataset = dset.MNIST('./', train=True, download=True)


'''デバイスの設定'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''初期設定'''

dir_path = "./drive/MyDrive/result"
os.makedirs(dir_path, exist_ok=True)


model_name = "SAGAN"
f_path_result = "./drive/MyDrive/result/{}".format(model_name)
f_path_params = "./drive/MyDrive/params/{}".format(model_name)

os.makedirs(f_path_result, exist_ok=True)
os.makedirs(f_path_params, exist_ok=True)


'''ハイパーパラメータの設定'''

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_epoch", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr_g", type=float, default=1e-4)
parser.add_argument("--lr_d", type=float, default=4e-4)
parser.add_argument("--nch_g", type=int, default=64) # 生成器のチャネル数
parser.add_argument("--nch_d", type=int, default=64) # 識別器のチャネル数
parser.add_argument("--z_dim", type=int, default=100) # ノイズの次元
parser.add_argument("--beta1", type=float, default=0.0) # Adamのハイパーパラメータ
opt = parser.parse_args(args=[])
print(opt)


'''データセットの設定'''

class ImageDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.file_list = glob.glob("/content/drive/MyDrive/data/102flowers/jpg/*.jpg")
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is not None:
            img = self.transform(Image.open(self.file_list[index]))
        else:
            img = Image.open(self.file_list[index])
        return img

    def __len__(self):
        return len(self.file_list)


'''Self-Attentionレイヤーの設定'''

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels // 8
        self.fx_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gx_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)        
        self.hx_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        B, C, H, W = x.size()
        fx = self.fx_1x1(x).view(B, -1, H * W).permute(0, 2, 1)
        gx = self.gx_1x1(x).view(B, -1, H * W)
        hx = self.hx_1x1(x).view(B, -1, H * W)
        s_mtx = torch.bmm(fx, gx)
        attention = self.softmax(s_mtx)
        o = torch.bmm(hx, attention)
        o = o.view(B, -1, H, W)
        out = x + self.gamma * o
        return out


'''生成器の定義'''

class Generator(nn.Module):
    
    def __init__(self, self_attention, z_dim=100, ngf=64, nc=3):
        super().__init__()
        self.convt1 = self.conv_trans_layers(z_dim, 8 * ngf, 4, 1, 0, True)
        self.convt2 = self.conv_trans_layers(8 * ngf, 4 * ngf, 4, 2, 1, True)
        self.convt3 = self.conv_trans_layers(4 * ngf, 2 * ngf, 4, 2, 1, True)
        self.attention1 = self_attention(2 * ngf)
        self.convt4 = self.conv_trans_layers(2 * ngf, ngf, 4, 2, 1, True)
        self.attention2 = self_attention(ngf)
        self.convt5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
        )

    @staticmethod
    def conv_trans_layers(in_channels, out_channels, kernel_size, stride, padding, has_norm):
        layers = [nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))]
        if has_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        net = nn.Sequential(*layers)
        return net
    
    def forward(self, x):
        out = self.convt1(x)
        out = self.convt2(out)
        out = self.convt3(out)
        out = self.attention1(out)
        out = self.convt4(out)
        out = self.attention2(out)
        out = self.convt5(out)
        return out


'''識別器の定義'''

class Discriminator(nn.Module):
    
    def __init__(self, self_attention, nc=3, ndf=64):
        super().__init__()
        self.conv1 = self.conv_layers(nc, ndf)
        self.conv2 = self.conv_layers(ndf, 2 * ndf)
        self.conv3 = self.conv_layers(2 * ndf, 4 * ndf)
        self.attention1 = self_attention(4 * ndf)
        self.conv4 = self.conv_layers(4 * ndf, 8 * ndf)
        self.attention2 = self_attention(8 * ndf)
        self.conv5 = nn.Conv2d(8 * ndf, 1, 4)

    @staticmethod
    def conv_layers(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                    has_batch_norm=False):
        layers = [nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))]
        if has_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        net = nn.Sequential(*layers)
        return net
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.attention1(out)
        out = self.conv4(out)
        out = self.attention2(out)
        out = self.conv5(out)
        return out


'''畳み込み層や BatchNormalizationを初期化するための関数の定義'''

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


'''前処理'''

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ),(0.5, ))
])


'''MNISTデータセットをダウンロード'''

dataset = ImageDataset(transform=transform)

dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size,shuffle=True)


'''netのインスタンス化'''

netG = Generator(SelfAttention, z_dim=opt.z_dim, ngf=opt.nch_g).to(device)
netG.apply(weights_init)
print(netG)

netD = Discriminator(SelfAttention, nc=3, ndf=opt.nch_d).to(device)
netD.apply(weights_init)
print(netD)


'''損失関数と最適化関数の定義'''

# 損失関数
criterion = nn.BCELoss()

# 最適化関数
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999), weight_decay=1e-5)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999), weight_decay=1e-5)


'''パラメーターを保存する関数'''

def save_params(file_path, epoch, netD, netG):
    torch.save(
        netG.state_dict(),
        file_path + "/g_{:04d}.pth".format(epoch)
    )

    torch.save(
        netD.state_dict(),
        file_path + "/d_{:04d}.pth".format(epoch)
    )


'''学習'''

lossesD = []
lossesG = []
raw_lossesD = []
raw_lossesG = []
relu = torch.nn.ReLU()

for epoch in range(opt.n_epoch):
    running_lossD = 0.0
    running_lossG = 0.0
    for i, real_imgs in enumerate(tqdm.tqdm(dataloader, position=0)):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size()[0]
        noise = torch.randn(batch_size, opt.z_dim, 1, 1).to(device)

        """Discriminatorの訓練"""
        
        netD.zero_grad()
        output = netD(real_imgs)
        lossD_real = torch.mean(relu(1.0 - output))

        fake_imgs = netG(noise)
        output = netD(fake_imgs.detach())
        lossD_fake = torch.mean(relu(1.0 + output))

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()
        
        """Generatorの訓練"""
        
        netG.zero_grad()
        output = netD(fake_imgs)

        lossG = -torch.mean(output)
        lossG.backward()
        optimizerG.step()

        running_lossD += lossD.item()
        running_lossG += lossG.item()
        raw_lossesD.append(lossD.item())
        raw_lossesG.append(lossG.item())
        
    running_lossD /= len(dataloader)
    running_lossG /= len(dataloader)
    print("epoch: {}, lossD: {}, lossG: {}".format(epoch, running_lossD, running_lossG))
    lossesD.append(running_lossD)
    lossesG.append(running_lossG)
    
    # 画像の表示
    grid_imgs = vutils.make_grid(fake_imgs[:24].detach() + 0.5)
    grid_imgs_arr = grid_imgs.cpu().numpy()
    plt.imshow(np.transpose(grid_imgs_arr, (1, 2, 0)))
    plt.show()
    
    # パラメータと画像の保存
    vutils.save_image(fake_imgs + 0.5, f_path_result + "/{}.jpg".format(epoch))
    save_params(f_path_params, epoch, netD, netG)


'''訓練イテレーションごとの損失'''

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(raw_lossesG,label="G")
plt.plot(raw_lossesD,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


'''エポックごとの損失'''

plt.figure(figsize=(10,5))
plt.title("Epoch_Loss")
plt.plot(lossesG,label="G")
plt.plot(lossesD,label="D")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()