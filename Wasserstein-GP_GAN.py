# ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

dir_path = "/content/drive/My Drive/Colab Notebooks/Pytorch-GANs-Practice/result"
os.makedirs(dir_path, exist_ok=True)


model_name = "WGAN-GP"
f_path_result = "/content/drive/My Drive/Colab Notebooks/Pytorch-GANs-Practice/result/{}".format(model_name)
f_path_params = "/content/drive/My Drive/Colab Notebooks/Pytorch-GANs-Practice/params/{}".format(model_name)

os.makedirs(f_path_result, exist_ok=True)
os.makedirs(f_path_params, exist_ok=True)


'''ハイパーパラメータの設定'''

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--nch_g", type=int, default=128) # 生成器のチャネル数
parser.add_argument("--nch_d", type=int, default=128) # 識別器のチャネル数
parser.add_argument("--z_dim", type=int, default=100) # ノイズの次元
parser.add_argument("--beta1", type=float, default=0.5) # Adamのハイパーパラメータ
parser.add_argument("--c_lower", type=float, default=-0.01)
parser.add_argument("--c_upper", type=float, default=0.01)
parser.add_argument("--n_critic", type=int, default=5)
parser.add_argument("--gp_weight", type=float, default=10.0)
opt = parser.parse_args(args=[])
print(opt)


'''生成器の定義'''

class Generator(nn.Module):

    def __init__(self, z_dim=100, ngf=128, nc=1): # nc: 色チャネル
        super().__init__()
        self.convt1 = self.conv_trans_layers(z_dim, 4 * ngf, 3, 1, 0)
        self.convt2 = self.conv_trans_layers(4 * ngf, 2 * ngf, 3, 2, 0)
        self.convt3 = self.conv_trans_layers(2 * ngf, ngf, 4, 2, 1)
        self.convt4 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
        )


    @staticmethod
    def conv_trans_layers(in_channels, out_channels, kernel_size, stride, padding):
        net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                               stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return net
    
    def forward(self, x):
        out = self.convt1(x)
        out = self.convt2(out)
        out = self.convt3(out)
        out = self.convt4(out)
        return out


'''識別器の定義'''

class Discriminator(nn.Module):

    def __init__(self, nc=1, ndf=128):
        super().__init__()
        self.conv1 = self.conv_layers(nc, ndf, has_batch_norm=False)
        self.conv2 = self.conv_layers(ndf, 2 * ndf, has_batch_norm=False)
        self.conv3 = self.conv_layers(2 * ndf, 4 * ndf, 3, 2, 0, has_batch_norm=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * ndf, 1, 3, 1, 0),
            nn.Sigmoid()
        )

    @staticmethod
    def conv_layers(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                    has_batch_norm=True):
        layers = [
                  nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                            padding, bias=False)
        ]
        if has_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        net = nn.Sequential(*layers)
        return net
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
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

dataset = dset.MNIST('./', train=True, download=True, transform=transform)

dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)


'''netのインスタンス化'''

netG = Generator(z_dim=opt.z_dim, ngf=opt.nch_g).to(device)
netG.apply(weights_init)
print(netG)

netD = Discriminator(nc=1, ndf=opt.nch_d).to(device)
netD.apply(weights_init)
print(netD)


'''損失関数と最適化関数の定義'''

# 損失関数
criterion = nn.BCELoss()

# 最適化関数
optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr, weight_decay=1e-4)
optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr, weight_decay=1e-4)


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


'''Gradient Penaltyを計算する関数'''

def gradient_penalty(real_imgs, fake_imgs, gp_weight, netD, device):
    batch_size = real_imgs.size()[0]
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_imgs).to(device)
    interpolated_imgs = (alpha * real_imgs.data + (1 - alpha) * fake_imgs.data).requires_grad_()
    interpolated_out = netD(interpolated_imgs)
    grad_outputs = torch.ones(interpolated_out.size()).to(device)
    gradients = torch.autograd.grad(interpolated_out, interpolated_imgs,
                                    grad_outputs=grad_outputs,
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch_size, -1)
    eps = 1e-12
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + eps)
    gp = gp_weight * ((gradients_norm - 1) ** 2).mean()
    return gp


'''学習'''

lossesD = []
lossesG = []
raw_lossesD = []
raw_lossesG = []

for epoch in range(opt.n_epoch):
    running_lossD = 0.0
    running_lossG = 0.0
    for i, (real_imgs, labels) in enumerate(tqdm.tqdm(dataloader, position=0)):
        
        '''weight clipping: 識別器'''
        
        '''
        for p in netD.parameters():
            p.data.clamp_(opt.c_lower, opt.c_upper)
        '''
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size()[0]
        noise = torch.randn(batch_size, opt.z_dim, 1, 1).to(device)

        shape = (batch_size, 1, 1, 1)
        labels_real = torch.ones(shape).to(device)
        labels_fake = torch.zeros(shape).to(device)
        
        
        '''識別器の訓練'''
        
        netD.zero_grad()
        output = netD(real_imgs)
        lossD_real = -torch.mean(output)
        # lossD_real = criterion(output, labels_real)

        fake_imgs = netG(noise)
        output = netD(fake_imgs.detach())
        lossD_fake = torch.mean(output)
        lossD_gp = gradient_penalty(real_imgs, fake_imgs, opt.gp_weight, netD, device)
        # lossD_fake = criterion(output, labels_fake)

        lossD = lossD_real + lossD_fake + lossD_gp
        lossD.backward()
        optimizerD.step()
        
        '''生成器の訓練'''
        
        if i % opt.n_critic == 0:
            netG.zero_grad()
            output = netD(fake_imgs)
            # lossG = criterion(output, labels_real)
            lossG = -torch.mean(output)
            lossG.backward()
            optimizerG.step()
            raw_lossesG.append(lossG.item())
        
        running_lossD += lossD.item()
        running_lossG += lossG.item()
        raw_lossesD.append(lossD.item())
        
    running_lossD /= len(dataloader)
    running_lossG /= len(dataloader)
    print("epoch: {}, lossD: {}, lossG: {}".format(epoch, running_lossD, running_lossG))
    
    lossesD.append(running_lossD)
    lossesG.append(running_lossG)
    
    '''フェイク画像の表示'''
    
    grid_imgs = vutils.make_grid(fake_imgs[:24].detach())
    grid_imgs_arr = grid_imgs.cpu().numpy()
    plt.imshow(np.transpose(grid_imgs_arr, (1, 2, 0)))
    plt.show()
    
    '''画像とパラメータの保存'''
    
    vutils.save_image(fake_imgs, f_path_result + "/{}.jpg".format(epoch))
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
plt.title("Epoch Loss")
plt.plot(lossesG,label="G")
plt.plot(lossesD,label="D")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()