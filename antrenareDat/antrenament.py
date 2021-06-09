from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

#-------- Reteaua neurala primita --------#
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Incercare de retea neurala (structura momentan nefunctionala)
class Net(nn.Module):

   def __init__(self):
       super(Net, self).__init__()
       self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = 5, stride = 1)
       self.conv2 = nn.Conv2d(in_channels = 20, out_channels = 50, kernel_size = 3, stride = 1)
       self.conv3 = nn.Conv2d(in_channels= 50, out_channels = 100, kernel_size = 3, stride=1)
       self.fc1 = nn.Linear(in_features = 8 * 8 * 100, out_features = 500)
       self.fc2 = nn.Linear(in_features = 500, out_features = 10)

   def forward(self, x):
       x = F.relu(self.conv1(x))
       x = F.max_pool2d(x, 2, 2)
       x = F.relu(self.conv2(x))
       x = F.max_pool2d(x, 2, 2)
       x = F.relu(self.conv3(x))
       print(x.shape)
       x = x.view(-1, 3 * 3 * 100)
       x = F.relu(self.fc1(x))
       x = self.fc2(x)

       return F.log_softmax(x, dim=1)

#-------- Functia de afisare a unei imagini --------#
def imshow(img, title):
        mean = 0.1307
        standard = 0.3081
        # img = img / 2 + 0.5  # unnormalize
        img = (img * standard) + mean  # unnormalize

        # 0.1307 mean
        # 0.3081 standard

        npimg = img.numpy()
        plt.title(title)
        plt.imshow((np.transpose(npimg, (1, 2, 0))))
        plt.show()
#---------------------------------------------------#

#-------- Functia de antrenare data a retelei neurale --------#
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
#------------------------------------------------------------#

#-------- Afisare imagini din baza de date ade antrenament  --------#
def imageShow(database, title):
    # for i, data, in enumerate(database):
    #     dataiter = iter(database)
    #     # get some random training images
    #     images, labels = dataiter.next()
    #     # show images
    #     imshow(torchvision.utils.make_grid(images), title)
    #     break
    dataiter = iter(database)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images[0:3]), title)

#-------------------------------------------------------------------#

#-------- Rezolvare punct 9  --------#
def imageShowLabels(database, title):
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    dataiter = iter(database)
    images, labels = dataiter.next()
    print('Label: '.join('%5s' % classes[labels[i]] for i in range(3))) # de rezolvat formatul la print cas a arat mai bine label-urile
    imshow(torchvision.utils.make_grid(images[0:3]), title)
    # print(' '.join('%5s' % classes[labels[j]] for j in range(3)))

#------------------------------------#

def main():
    # Training settings ? De ce a folosit asa sa transfere argumentele?
    # ?Cum functioneaza parser-ul? De ce nu au fost stocate in niste variabile ci au fost folosite argumente?

    # Argumente folosite pentru algoritmii de antrenare si testare date
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch-size-img', type=int, default=3, metavar='N',
                        help='input training images to show to user (default: 3)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)') # ? Ce reprezinta seed-ul?
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu") # Incarcarea retelei neurale pe placa video sau pe cpu

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Incarcare baza de date de antrenament
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = Net().to(device)  # Instantiere RN pe dispozitivul ales
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) # Instantiere opimizator

    #----------------- Rezolvare pct 7 ------------------#
    # Afisarea a 3 imagini din baza de date de antrenament
    imageShowLabels(train_loader, "Training-images")
    #----------------------------------------------------#


    # Antrenarea si testarea propriu-zisa a retelei neurale
    # for epoch in range(1, args.epochs + 1):
    #    train(args, model, device, train_loader, optimizer, epoch)



if __name__ == '__main__':
    main()