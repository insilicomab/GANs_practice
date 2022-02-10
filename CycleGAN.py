# ライブラリのインポート
import itertools
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import os

import torch
import tqdm
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


'''デバイスの設定'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''初期設定'''

model_name = "CycleGAN"
f_path_result = "./drive/MyDrive/result/{}".format(model_name)
f_path_params = "./drive/MyDrive/params/{}".format(model_name)

os.makedirs(f_path_result, exist_ok=True)
os.makedirs(f_path_params, exist_ok=True)


'''ハイパーパラメータの設定'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_epoch", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--decay_start", type=int, default=100)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--weight_identity", type=float, default=5.0)
parser.add_argument("--weight_cycle", type=float, default=10.0)
parser.add_argument("--beta1", type=float, default=0.5)
opt = parser.parse_args(args=[])
print(opt)


'''データセットの設定'''

class ImageDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.files_A = glob.glob("./drive/MyDrive/data/horse2zebra/trainA/*.jpg")
        self.files_B = glob.glob("./drive/MyDrive/data/horse2zebra/trainB/*.jpg")
        self.transform = transform

    def __getitem__(self, index):
        imgA = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        while True:
            random_index = np.random.randint(0, len(self.files_B) - 1) # ランダムなインデックスを取得
            imgB = self.transform(Image.open(self.files_B[random_index % len(self.files_B)]))
            C, H, W = imgB.size()
            if C == 3:
                break
        return {"A": imgA, "B":imgB}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class DecayLR(object):
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    
    def step(self, epoch):
        # 100エポックまでが一定でその後、減少
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
    

'''learning_rateのスケジューラの設定'''

class ReplayBuffer(object):
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if np.random.rand() > 0.5:
                    i = np.random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


'''50枚の画像をBuffer層にためていく'''

class ReplayBuffer(object):
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = [] # dataにフェイクの画像を貯める

    def push_and_pop(self, data):
        to_return = [] # 返ってくる画像
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            # max_sizeになるまでto_returnに画像をためていく
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if np.random.rand() > 0.5:
                    i = np.random.randint(0, self.max_size - 1) # 0-49
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


'''ResidualBlockの実装'''

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out += x
        return out


'''Generatorの実装'''

class Generator(nn.Module):
    def __init__(self, res_block, in_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.res_block = res_block(256)
        self.transformer = nn.ModuleList([
            res_block(256),
            res_block(256),
            res_block(256),
            res_block(256),
            res_block(256),
            res_block(256),
            res_block(256),
            res_block(256),
            res_block(256)
        ])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.encoder(x)
        for func in self.transformer:
            out = func(out)
        out = self.decoder(out)
        return out


'''Discriminatorの実装'''

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_layer(3, 64, 4, 2, 1, False)
        self.conv2 = self.conv_layer(64, 128, 4, 2, 1, True)
        self.conv3 = self.conv_layer(128, 256, 4, 2, 1, True)
        self.conv4 = self.conv_layer(256, 512, 4, 1, 1, True)
        self.conv5 = nn.Conv2d(512, 1, 4, 1, padding=1)


    @staticmethod
    def conv_layer(in_channels, out_channels, kernel_size, stride, padding, has_norm=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if has_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        net = nn.Sequential(*layers)
        return net

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        B, C, H, W = out.size()
        out = F.avg_pool2d(out, (H, W))
        out = out.view(B, -1)
        return out


'''畳み込み層や BatchNormalizationを初期化するための関数の定義'''

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


'''transformの設定'''

transform = transforms.Compose([
    transforms.Resize(int(opt.image_size * 1.12), Image.BICUBIC),
    transforms.RandomCrop(opt.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))                                
])


'''DataLoader'''

dataset = ImageDataset(transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size,
                        shuffle=True)


'''GeneratorとDiscriminatorのインスタンス化'''

netG_A2B = Generator(ResidualBlock).to(device) # 馬からシマウマへ
netG_B2A = Generator(ResidualBlock).to(device) # シマウマから馬へ
netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

netG_A2B.apply(weights_init)
netG_B2A.apply(weights_init)
netD_A.apply(weights_init)
netD_B.apply(weights_init)


'''損失関数と最適化関数の定義'''

# 損失関数
adversarial_loss = torch.nn.MSELoss().to(device)
cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)

# 最適化関数
optimizer_D_A = optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_D_B = optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_G = optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                         lr=opt.lr, betas=(opt.beta1, 0.999))


'''learning_rateのスケジューラの設定'''

lr_lambda = DecayLR(opt.n_epoch, 0, opt.decay_start).step
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lr_lambda)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lr_lambda)


'''パラメーターを保存する関数'''

def save_params(epoch, dir_path, model_list, model_name_list):
    for model, model_name in zip(model_list, model_name_list):
        file_path = dir_path + "/{model}_{epoch}.pth".format(model=model_name, epoch=epoch)
        torch.save(model.state_dict(), file_path)


'''学習'''

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

for epoch in range(118, opt.n_epoch):
    running_loss_D = 0.0
    running_loss_G = 0.0
    running_loss_G_GAN = 0.0
    running_loss_G_cycle = 0.0
    running_loss_G_identity = 0.0
    for data in tqdm.tqdm(dataloader, position=0):
        real_img_A = data["A"].to(device)
        real_img_B = data["B"].to(device)
        batch_size = real_img_A.size()[0]
        real_label = torch.ones([batch_size, 1]).to(device)
        fake_label = torch.zeros([batch_size, 1]).to(device)
        
        """Generatorの訓練"""
        
        optimizer_G.zero_grad()
        
        """adversarial loss"""
        
        fake_img_A = netG_B2A(real_img_B)
        fake_img_B = netG_A2B(real_img_A)
        output_A = netD_A(fake_img_A)
        output_B = netD_B(fake_img_B)

        loss_GAN_A2B = adversarial_loss(output_B, real_label)
        loss_GAN_B2A = adversarial_loss(output_A, real_label)
        
        """cycle loss"""
        
        cycle_img_A = netG_B2A(fake_img_B)
        cycle_img_B = netG_A2B(fake_img_A)

        loss_cycle_ABA = cycle_loss(cycle_img_A, real_img_A)
        loss_cycle_BAB = cycle_loss(cycle_img_B, real_img_B)
        
        """identity loss"""
        
        identity_img_A = netG_B2A(real_img_A)
        identity_img_B = netG_A2B(real_img_B)
        loss_identity_A = identity_loss(identity_img_A, real_img_A)
        loss_identity_B = identity_loss(identity_img_B, real_img_B)

        lossG = (loss_GAN_A2B + loss_GAN_B2A + 
                 opt.weight_identity * (loss_identity_A + loss_identity_B ) + opt.weight_cycle * (loss_cycle_ABA + loss_cycle_BAB))
        lossG.backward()
        optimizer_G.step()

        """Discriminatorの訓練"""
        
        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()

        real_output_A = netD_A(real_img_A)
        real_output_B = netD_B(real_img_B)
        loss_DA_real = adversarial_loss(real_output_A, real_label)
        loss_DB_real = adversarial_loss(real_output_B, real_label)

        fake_img_A = fake_A_buffer.push_and_pop(fake_img_A)
        fake_img_B = fake_B_buffer.push_and_pop(fake_img_B)
        fake_output_A = netD_A(fake_img_A.detach())
        fake_output_B = netD_B(fake_img_B.detach())

        loss_DA_fake = adversarial_loss(fake_output_A, fake_label)
        loss_DB_fake = adversarial_loss(fake_output_B, fake_label)

        loss_DA = (loss_DA_real + loss_DA_fake) * 0.5
        loss_DB = (loss_DB_real + loss_DB_fake) * 0.5

        loss_DA.backward()
        loss_DB.backward()
        optimizer_D_A.step()
        optimizer_D_B.step()
        
        running_loss_D += (loss_DA.item() + loss_DB.item()) / 2.0
        running_loss_G += lossG.item()
        running_loss_G_GAN += (loss_GAN_A2B.item() + loss_GAN_B2A.item()) / 2.0
        running_loss_G_cycle += (loss_cycle_ABA.item() + loss_cycle_BAB.item()) / 2.0
        running_loss_G_identity += (loss_identity_A.item() + loss_identity_B.item()) / 2.0
        
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    running_loss_D /= len(dataloader)
    running_loss_G /= len(dataloader)
    running_loss_G_GAN /= len(dataloader)
    running_loss_G_cycle /= len(dataloader)
    running_loss_G_identity /= len(dataloader)
    loss_log = """epoch: {}, Loss D: {}, Loss G GAN: {},
     Loss G cycle: {}, Loss G identity: {}""".format(epoch, running_loss_D, running_loss_G,
                                                     running_loss_G_GAN, running_loss_G_cycle, running_loss_G_identity)
    print(loss_log)
    fake_imgs = torch.cat([fake_img_A, fake_img_B])
    grdi_imgs = vutils.make_grid(fake_imgs.detach())
    grdi_igms_arr = grdi_imgs.cpu().numpy()
    plt.imshow(np.transpose(grdi_igms_arr + 0.5, (1, 2, 0)))
    plt.show()
    model_list = [netG_A2B, netG_B2A, netD_A, netD_B]
    model_name_list = ["netG_A2B", "netG_B2A", "netD_A", "netD_B"]
    save_params(epoch, f_path_params, model_list, model_name_list)