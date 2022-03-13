# +
import torch


import utils
import argparse
import settings
from utils import get_training_dataloader, get_training_dataloader_LRP, get_test_dataloader,get_test_dataloader_LRP, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
import torch.nn.functional as F
import torch.nn as nn

# +
import random
import torch.backends.cudnn as cudnn
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
# -

import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "1"

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18', help='net type')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-pretrain', action='store_false', default=False, help='pretrain')
    args = parser.parse_args()
except:
    args = parser.parse_args(args=[])

args

model = utils.get_model(args.net, pretrain = args.pretrain)
model = model.cuda()

# +
cifar100_training_loader_LRP = get_training_dataloader_LRP(
    settings.CIFAR100_TRAIN_MEAN,
    settings.CIFAR100_TRAIN_STD,
    num_workers=4,
    batch_size=args.b,
    shuffle=True
)

cifar100_test_loader_LRP = get_test_dataloader_LRP(
    settings.CIFAR100_TRAIN_MEAN,
    settings.CIFAR100_TRAIN_STD,
    num_workers=4,
    batch_size=args.b,
    shuffle=True
)

cifar100_test_loader = get_test_dataloader(
    settings.CIFAR100_TRAIN_MEAN,
    settings.CIFAR100_TRAIN_STD,
    num_workers=4,
    batch_size=args.b,
    shuffle=True
)
# -

import torch.nn as nn
import torch.optim as optim

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
iter_per_epoch = len(cifar100_training_loader_LRP)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)


# +
import os
if not os.path.exists(settings.CHECKPOINT_PATH):
    os.mkdir(settings.CHECKPOINT_PATH)

if args.pretrain:
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net + "_Pretrain_LRP", settings.TIME_NOW)
else:
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net + "_LRP", settings.TIME_NOW)


# -

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
file = open(checkpoint_path + "/log.txt", "w")
checkpoint_path = os.path.join(checkpoint_path, '{model}-{epoch}-{type}.pth')

import loss

loss_function=loss.RefineryLoss()


# +
import time
best_acc = 0.0
accs = []
for epoch in range(1, settings.EPOCH + 1):
    if epoch > args.warm:
        train_scheduler.step(epoch)

    model.train()
    utils.train_LRP(model,cifar100_training_loader = cifar100_training_loader_LRP, optimizer = optimizer,
                    epoch = epoch, loss_function= loss_function, file = file, args = args, warmup_scheduler = warmup_scheduler)
    model.eval()
#     acc = utils.eval_training(model, cifar100_test_loader = cifar100_test_loader, optimizer =  optimizer, 
#                             loss_function = nn.CrossEntropyLoss(), epoch= epoch,  file =file, args = args)

    acc = utils.eval_training_LRP(model, cifar100_test_loader = cifar100_test_loader_LRP, optimizer =  optimizer, 
                            loss_function = loss_function, epoch= epoch,  file =file, args = args)    
    accs.append(acc.item())
    file.flush()
    #start to save best performance model after learning rate decay to 0.01
    if epoch > settings.MILESTONES[1] and best_acc < acc:
        best_acc = acc
        continue


file.close()
# -
from matplotlib import pyplot as plt
plt.plot(accs)
plt.show()

for d in cifar100_training_loader_LRP:
    print(d[0].shape)
    break


def format_change(arr):
    return torch.cat((arr[0].unsqueeze(-1), arr[1].unsqueeze(-1), arr[2].unsqueeze(-1)), dim = -1)


format_change(d[0][0]).shape

# +
# import torchvision
# import torchvision.transforms as transforms

# transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize(size=(256, 256)),
#             transforms.RandomCrop(200),
# ])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform = transform)

# +
# plt.imshow(format_change(trainset[0][0]))

# +
# for d in cifar100_test_loader:
#     print(d[0].shape)
#     break

# +
# from torch.optim.lr_scheduler import _LRScheduler
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from glob import glob
# import torchvision
# import datetime
# import numpy
# import torch
# import pickle
# import time
# import sys
# import os
# import re

# import torch.nn.functional as F


# +
# class Cifar100(Dataset):
#     def __init__(self, data_dir, transform = None):
#         self.transform = transform
#         self.train_data = []
#         self.train_files = glob(data_dir + "/*.pickle")


#     def __len__(self):
#         return len(self.train_files)

#     def __getitem__(self,idx):
#         with open(self.train_files[idx], 'rb') as f:
#             data = pickle.load(f)
            
#         img = self.transform(torch.tensor(data['img']))
#         return img, data['softlabel'], data['label']


# +
# data_dir = "LRP_data_resnext101_32x8d_0_8514/train/"
# transform_train = transforms.Compose([
# #     transforms.ToPILImage(),
# #     transforms.ToTensor(),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD),
#     transforms.Resize(size=(224, 224)),
# ])


# cifar100_training = Cifar100(transform=transform_train, data_dir = data_dir)

# +
# cifar100_training[0][0].shape
