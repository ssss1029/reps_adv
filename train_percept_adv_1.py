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

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='allconv',
                    choices=['allconv'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./checkpoints/TEMP', help='Folder to save checkpoints.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
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

# # mean and standard deviation of channels of CIFAR-10 images
# mean = [x / 255 for x in [125.3, 123.0, 113.9]]
# std = [x / 255 for x in [63.0, 62.1, 66.7]]
normalize = trn.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))

train_transform = trn.Compose([
    trn.RandomHorizontalFlip(), 
    trn.RandomCrop(32, padding=4),
    trn.ToTensor(),
    normalize
])

test_transform = trn.Compose([
    trn.ToTensor(),
    normalize
])

if args.dataset == 'cifar10':
    train_data = dset.CIFAR10('/data/sauravkadavath/cifar10-dataset/', train=True, transform=train_transform)
    test_data = dset.CIFAR10('/data/sauravkadavath/cifar10-dataset/', train=False, transform=test_transform)
    num_classes = 10
else:
    train_data = dset.CIFAR100('/data/sauravkadavath/cifar10-dataset/', train=True, transform=train_transform)
    test_data = dset.CIFAR100('/data/sauravkadavath/cifar10-dataset/', train=False, transform=test_transform)
    num_classes = 100


train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
if args.model == 'allconv':
    net = AllConvNet(num_classes).cuda()
# else:
#     net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

start_epoch = 0

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))

adversary = attacks.PGD(epsilon=8./255, num_steps=3, step_size=2./255).cuda()

def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    for bx, by in tqdm(train_loader):

        bx, by, = bx.cuda(), by.cuda()

        adv_bx = adversary(net, bx, by)

        # forward
        logits = net(adv_bx)

        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss = F.cross_entropy(logits, by)
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.9 + float(loss) * 0.1

    state['train_loss'] = loss_avg

# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + args.model +
                                  '_baseline_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + args.model +
                            '_baseline_epoch_' + str(epoch) + '.pt'))
    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.dataset + args.model +
                             '_baseline_epoch_' + str(epoch - 1) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results

    with open(os.path.join(args.save, args.dataset + args.model +
                                      '_baseline_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )
