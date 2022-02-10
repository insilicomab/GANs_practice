'''ハイパーパラメータの設定'''

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_epoch", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--nch_g", type=int, default=128) # 生成器のチャネル数
parser.add_argument("--nch_d", type=int, default=128) # 識別器のチャネル数
parser.add_argument("--z_dim", type=int, default=100) # ノイズの次元
parser.add_argument("--beta1", type=float, default=0.5) # Adamのハイパーパラメータ
opt = parser.parse_args(args=[])
print(opt)


'''Datasetの定義'''

# ライブラリのインポート
from torch.utils.data import Dataset
from　PIL import Image


class ImageDataset(Dataset):
    
    def __init__(self, file_list, transform=None):
        super().__init__()
        self.file_list = file_list
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if self.transform is not None:
            img = self.transform(Image.open(self.file_list[index]))
        else:
            img = Image.open(self.file_list[index])
        return img


'''前処理'''

import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ),(0.5, ))
])
    

'''MNISTデータセットをダウンロード'''

import torchvision.datasets as dset

dataset = dset.MNIST('./', train=True, download=True)


'''DataLoaderの定義'''

from torch.utils.data import DataLoader(dataset=dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=True
                                        )


'''畳み込み層や BatchNormalizationを初期化するための関数の定義'''

import torch.nn as nn

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)