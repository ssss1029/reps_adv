# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.allconv import AllConvNet
import attacks.attacks_percept as attacks

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='allconv', choices=['allconv'], help='Choose architecture.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--save', '-s', type=str, default='./images_saved/TEMP', help='Folder to save checkpoints.')
parser.add_argument('--load', type=str, default=None, required=True, help='Model to load.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

if os.path.exists(args.save):
    resp = "None"
    while resp.lower() not in {'y', 'n'}:
        resp = input("Save directory {0} exits. Continue? [Y/n]: ".format(args.save))
        if resp.lower() == 'y':
            break
        elif resp.lower() == 'n':
            exit(1)
        else:
            pass

torch.manual_seed(1)
np.random.seed(1)

normalize = trn.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))

test_transform = trn.Compose([
    trn.ToTensor(),
    normalize
])

if args.dataset == 'cifar10':
    test_data = dset.CIFAR10('/data/sauravkadavath/cifar10-dataset/', train=False, transform=test_transform)
    num_classes = 10
else:
    test_data = dset.CIFAR100('/data/sauravkadavath/cifar10-dataset/', train=False, transform=test_transform)
    num_classes = 100


test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
if args.model == 'allconv':
    net = AllConvNet(num_classes).cuda()

net.load_state_dict(torch.load(args.load))

adversary = attacks.PGD(epsilon=8./255, num_steps=10, step_size=0.5/255).cuda()

# def train():
#     net.train()  # enter train mode
#     loss_avg = 0.0
#     for bx, by in tqdm(train_loader):

#         bx, by, = bx.cuda(), by.cuda()

#         adv_bx = adversary(net, bx, by)

#         # print(torch.max(bx), torch.min(bx), torch.mean(bx))

#         # forward
#         logits = net(adv_bx)

#         # backward
#         scheduler.step()
#         optimizer.zero_grad()
#         loss = F.cross_entropy(logits, by)
#         loss.backward()
#         optimizer.step()

#         # exponential moving average
#         loss_avg = loss_avg * 0.9 + float(loss) * 0.1

#     state['train_loss'] = loss_avg

# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.cuda(), target.cuda()

            adv_data = adversary(net, data, target)

            # forward
            output = net(adv_data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)

    print("TEST LOSS = {0}".format(state['test_loss']))
    print("TEST ACC = {0}".format(state['test_accuracy']))


test()