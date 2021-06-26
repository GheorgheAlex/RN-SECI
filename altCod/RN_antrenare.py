from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F # import layerele
import torch.optim as optim # optimizare
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import random

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1, 2) # ordine in_features, out_features, kernel_size, stride, padding
        self.conv1_batch = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.conv2_batch = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 70, 2,2,1)
        self.conv3_batch = nn.BatchNorm2d(70)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3 * 3 * 70, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 4, 2)
        x = self.conv1_batch(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2, 1)
        x = self.conv2_batch(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2, 2, 1)
        x = F.max_pool2d(x, 1, 1)
        x = self.conv3_batch(x)
        x = x.view(-1, 3 * 3 * 70)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def imshow(img, title):
    mean = 0.1307
    standard = 0.3081
    img = (img * standard) + mean  # unnormalize
    npimg = img.numpy()
    plt.title(title)
    plt.imshow((np.transpose(npimg, (1, 2, 0))))
    plt.show()

# PUNCT 7
def imaginiAntrenament(database, title, batch_size):
    randomNumber0 = random.randint(0, batch_size)
    randomNumber1 = random.randint(0, batch_size)
    randomNumber2 = random.randint(0, batch_size)
    databaseIter = iter(database)
    images, labels = databaseIter.next()
    imshow(images[randomNumber0], title)
    imshow(images[randomNumber1], title)
    imshow(images[randomNumber2], title)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):  # batch-se face media si variatia datelor pt a putea fi normalizate
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)  # conversie din tensor in array
        # pycharm tensor to ndarray(tensorul din data trebe sa-l facem 28,28,1)
        # din data [64,2,28,28] luam de ex data[0] -?[1,28,28] -> [28,28,1] ->numpy  ndarray[28,28,1] ->im show sasu imwrite
        #                                 data[1]
        #                                 data[2]
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        # o epoca se termina atunci cand trecem o data prin toata multimea de antrenare
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',  # lr-learning rate
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training') # CUDA e de la NVIDIA si insemna ca foloseste placa video dedicata
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        # log-interval-la ce interval se face debug
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,  # daca se face sau nu save la model
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))  # preprocesarea-normalizare
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)


    model = Net().to(device)  # se declara un model de tip net si se pune pe device
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  # se declara un optimizator

    imaginiAntrenament(train_loader,"Imagine antrenament", args.batch_size)

    for epoch in range(1, args.epochs + 1):  # pt fiecare epoca se declara train si test
        train(args, model, device, train_loader, optimizer, epoch)

    if (args.save_model): # dupa antrenare se salveaza modelul
        torch.save(model.state_dict(), "mnist_cnn.pt")
        print("Am salvat modelul gata antrenat")


if __name__ == '__main__':
    main()