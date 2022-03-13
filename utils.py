# +
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from glob import glob
import torchvision
import datetime
import numpy
import torch
import pickle
import time
import sys
import os
import re

import torch.nn.functional as F


# +
import torchvision.models as models


def get_model(model_name, pretrain=True):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrain)
        model.fc = nn.Linear(512, 100)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrain)
        model.fc = nn.Linear(512, 100)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrain)
        model.fc = nn.Linear(2048, 100)

    elif model_name == "resnet101":
        model = models.resnet101(pretrained=pretrain)
        model.fc = nn.Linear(2048, 100)

    elif model_name == "resnet152":
        model.fc = nn.Linear(2048, 100)
        model = models.resnet152(pretrained=pretrain)
        
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=pretrain)
        model.classifier[6] = nn.Linear(4096, 100)

    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=pretrain)
        model.classifier[1] = nn.Conv2d(512, 100, kernel_size=(1,1))

    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrain)
        model.classifier[6] = nn.Linear(4096, 100)

    elif model_name == "densenet161":
        model = models.densenet161(pretrained=pretrain)
        model.classifier = nn.Linear(2208, 100)
        
    elif model_name == "inception_v3":
        model = models.inception_v3(pretrained=pretrain)
        
    elif model_name == "googlenet":
        model = models.googlenet(pretrained=pretrain)
        model.fc = nn.Linear(1024, 100)

    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(pretrained=pretrain)
        model.fc = nn.Linear(1024, 100)

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrain)
        model.classifier[1] = nn.Linear(1280, 100)

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=pretrain)
        model.classifier[3] = nn.Linear(1280, 100)

    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=pretrain)
        model.classifier[3] = nn.Linear(1024, 100)
        
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=pretrain)
        model.fc= nn.Linear(2048, 100)

    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=pretrain)
        model.fc= nn.Linear(2048, 100)

    elif model_name == "mnasnet1_0":
        model = models.mnasnet1_0(pretrained=pretrain)
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrain)
        
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(pretrained=pretrain)
        
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(pretrained=pretrain)
        
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=pretrain)
        
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(pretrained=pretrain)
        
    elif model_name == "efficientnet_b5":
        model = models.efficientnet_b5(pretrained=pretrain)
        
    elif model_name == "efficientnet_b6":
        model = models.efficientnet_b6(pretrained=pretrain)

    elif model_name == "efficientnet_b7":
        model = models.efficientnet_b7(pretrained=pretrain)

    elif model_name == "regnet_y_400mf":
        model = models.regnet_y_400mf(pretrained=pretrain)

    elif model_name == "regnet_y_800mf":
        model = models.regnet_y_800mf(pretrained=pretrain)

    elif model_name == "regnet_y_1_6gf":
        model = models.regnet_y_1_6gf(pretrained=pretrain)

    elif model_name == "regnet_y_3_2gf":
        model = models.regnet_y_3_2gf(pretrained=pretrain)

    elif model_name == "regnet_y_8gf":
        model = models.regnet_y_8gf(pretrained=pretrain)

    elif model_name == "regnet_y_16gf":
        model = models.regnet_y_16gf(pretrained=pretrain)

    elif model_name == "regnet_y_32gf":
        model = models.regnet_y_32gf(pretrained=pretrain)

    elif model_name == "regnet_x_400mf":
        model = models.regnet_x_400mf(pretrained=pretrain)

    elif model_name == "regnet_x_800mf":
        model = models.regnet_x_800mf(pretrained=pretrain)

    elif model_name == "regnet_x_1_6gf":
        model = models.regnet_x_1_6gf(pretrained=pretrain)

    elif model_name == "regnet_x_3_2gf":
        model = models.regnet_x_3_2gf(pretrained=pretrain)

    elif model_name == "regnet_x_8gf":
        model = models.regnet_x_8gf(pretrained=pretrain)

    elif model_name == "regnet_x_16gf":
        model = models.regnet_x_16gf(pretrainedpretrain)

    elif model_name == "regnet_x_32gf":
        model = models.regnet_x_32gf(pretrained=pretrain)

    elif model_name == "vit_b_16":
        model = models.vit_b_16(pretrained=pretrain)

    elif model_name == "vit_b_32":
        model = models.vit_b_32(pretrained=pretrain)

    elif model_name == "vit_l_16":
        model = models.vit_l_16(pretrained=pretrain)

    elif model_name == "vit_l_32":
        model = models.vit_l_32(pretrained=pretrain)

    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(pretrained=pretrain)

    elif model_name == "convnext_small":
        model = models.convnext_small(pretrained=pretrain)

    elif model_name == "convnext_base":
        model = models.convnext_base(pretrained=pretrain)

    elif model_name == "convnext_large":
        model = models.convnext_large(pretrained=pretrain)
    else:
        assert False, "Model Name is Wrong"

    return model



# -

def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Normalize(mean, std),
        transforms.Resize(size=(256, 256)),
        transforms.RandomCrop(224, padding=4),
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(224, 224)),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


class Cifar100(Dataset):
    def __init__(self, data_dir, transform = None):
        self.transform = transform
        self.train_data = []
        self.train_files = glob(data_dir + "/*.pickle")


    def __len__(self):
        return len(self.train_files)

    def __getitem__(self,idx):
        with open(self.train_files[idx], 'rb') as f:
            data = pickle.load(f)
        img = self.transform(torch.tensor(data['img']))
        
        return img, data['softlabel'], data['label']


def get_training_dataloader_LRP(mean, std, batch_size=16, num_workers=2, shuffle=True, use_LRP_image = False,
                               data_dir = "LRP_data_resnext101_32x8d_0_8514/train/"):


    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
#         transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize(size=(256, 256)),
        transforms.RandomCrop(224, padding=4),
        
    ])
    
    cifar100_training = Cifar100(transform=transform_train, data_dir = data_dir)
    print(len(cifar100_training))
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_test_dataloader_LRP(mean, std, batch_size=16, num_workers=2, shuffle=True, use_LRP_image = False,
                            data_dir = "LRP_data_resnext101_32x8d_0_8514/test/"):


    transform_test = transforms.Compose([
#         transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    
    cifar100_test = Cifar100(transform=transform_test, data_dir = data_dir)
    print(len(cifar100_test))
    cifar100_training_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


# +
def train_LRP(model, cifar100_training_loader, epoch, loss_function, args, optimizer, warmup_scheduler, file = None):

    start = time.time()
    model.train()
    for batch_index, (images, labels, real_labels) in enumerate(cifar100_training_loader):

        labels = labels.cuda()
        real_labels = real_labels.cuda()
        images = images.cuda()


        optimizer.zero_grad()
        outputs = model(images)
        labels = F.softmax(labels, dim=1)
         
        loss = loss_function((outputs, labels), real_labels.unsqueeze(-1))[0]
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(model.children())[-1]

        if batch_index % 100 == 0:
            log = 'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            )
            print(log)
            
            if file is not None:
                file.write(log)
                file.write("\n")
            


        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    if file is not None:
        file.write('epoch {} training time consumed: {:.2f}s\n'.format(epoch, finish - start))
        file.flush()
        
        
def eval_training_LRP(model, cifar100_test_loader, loss_function, args, optimizer,epoch=0, tb=True, file= None):

    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    with torch.no_grad():
        for (images, labels, real_labels) in cifar100_test_loader:

            images = images.cuda()
            labels = labels.cuda()
            labels = F.softmax(labels, dim=1)
            real_labels = real_labels.cuda()
            
            outputs = model(images)
            loss = loss_function((outputs, labels), real_labels.unsqueeze(-1))[0]

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(real_labels).sum()

        finish = time.time()

        print('Evaluating Network.....')
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(cifar100_test_loader.dataset),
            correct.float() / len(cifar100_test_loader.dataset),
            finish - start
        ))
        print()
        if file:
            file.write('Evaluating Network.....\n')
            file.write('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s\n\n'.format(
                epoch,
                test_loss / len(cifar100_test_loader.dataset),
                correct.float() / len(cifar100_test_loader.dataset),
                finish - start
            ))
        #add informations to tensorboard

        return correct.float() / len(cifar100_test_loader.dataset)


# +
def train(model, cifar100_training_loader, epoch, optimizer, loss_function, args, warmup_scheduler, file = None):

    start = time.time()
    model.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        labels = labels.cuda()
        images = images.cuda()


        optimizer.zero_grad()
        outputs = model(images)
         
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(model.children())[-1]

        if batch_index % 100 == 0:
            log = 'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            )
            print(log)
            
            if file is not None:
                file.write(log)
                file.write("\n")
            


        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    if file is not None:
        file.write('epoch {} training time consumed: {:.2f}s\n'.format(epoch, finish - start))
        file.flush()
        
        
def eval_training(model, cifar100_test_loader, loss_function,args,  optimizer,epoch=0, tb=True, file= None):

    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    with torch.no_grad():
        for (images, labels) in cifar100_test_loader:
            images = images.cuda()
            labels = labels.cuda()
            
            outputs = model(images)
            loss = loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

        finish = time.time()

        print('Evaluating Network.....')
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(cifar100_test_loader.dataset),
            correct.float() / len(cifar100_test_loader.dataset),
            finish - start
        ))
        print()
        if file:
            file.write('Evaluating Network.....\n')
            file.write('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s\n\n'.format(
                epoch,
                test_loss / len(cifar100_test_loader.dataset),
                correct.float() / len(cifar100_test_loader.dataset),
                finish - start
            ))
        #add informations to tensorboard

        return correct.float() / len(cifar100_test_loader.dataset)
