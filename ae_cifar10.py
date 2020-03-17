
"""

COPIED FROM
https://github.com/jellycsc/PyTorch-CIFAR-10-autoencoder/blob/master/main.py

"""

# Numpy
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Torchvision
import torchvision
import torchvision.transforms as transforms

# Matplotlib
import matplotlib.pyplot as plt

# OS
import os
import argparse

from tqdm import tqdm

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model():
    autoencoder = Autoencoder()
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
  			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False, help="Perform validation only.")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs.')
    parser.add_argument('--learing_rate', type=float, default=0.01, help='LR.')
    parser.add_argument('--save', type=str, default="checkpoints_ae/TEMP", help='Save directory.')
    parser.add_argument('--load', type=str, default=None, help='Load directory.')

    args = parser.parse_args()

    print(vars(args))

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
    else:
        os.makedirs(args.save, exist_ok=True)


    # Create model
    autoencoder = create_model()

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='/data/sauravkadavath/cifar10-dataset/', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.CIFAR10(root='/data/sauravkadavath/cifar10-dataset/', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.load:
        print("Loading checkpoint...")
        autoencoder.load_state_dict(torch.load(os.path.join(args.load, "ae.pt")))
    
    # TODO: Fix this.
    # if args.valid:
    #     dataiter = iter(testloader)
    #     images, labels = dataiter.next()
    #     imshow(torchvision.utils.make_grid(images))

    #     images = Variable(images.cuda())

    #     decoded_imgs = autoencoder(images)[1]
    #     imshow(torchvision.utils.make_grid(decoded_imgs.data))

    #     exit(0)

    # Define an optimizer and criterion
    criterion = nn.BCELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(tqdm(trainloader)):
            inputs = get_torch_vars(inputs)

            # ============ Forward ============
            encoded, outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs) 

            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ Logging ============
            running_loss += loss.data
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        
        torch.save(autoencoder.state_dict(), os.path.join(args.save, "ae.pt"))

        test(testloader, autoencoder, criterion)

    print('Finished Training')


def test(loader, autoencoder, criterion):

    with torch.no_grad():
        running_loss = 0
        for i, (inputs, _) in enumerate(loader):
            inputs = get_torch_vars(inputs)

            # ============ Forward ============
            encoded, outputs = autoencoder(inputs)

            if i == 0:
                torchvision.utils.save_image(inputs[:5], "images_saved/ae/ae_inputs_example.png")
                torchvision.utils.save_image(outputs[:5], "images_saved/ae/ae_outputs_example.png")

            loss = criterion(outputs, inputs) 

            # ============ Logging ============
            running_loss += loss.data
        
        print("Total Loss = ", running_loss)


if __name__ == '__main__':
    main()
