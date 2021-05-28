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

## Retea initiala

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

# Generarea retelei neurale
# class Net(nn.Module):
#     # Definirea straturilor retelei
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = 5, stride = 1)
#         self.conv2 = nn.Conv2d(in_channels = 20, out_channels = 50, kernel_size = 3, stride = 1)
#         self.conv3 = nn.Conv2d(in_channels= 50, out_channels = 100, kernel_size = 2, stride=1)
#         self.fc1 = nn.Linear(in_features = 4 * 4 * 100, out_features = 500)
#         self.fc2 = nn.Linear(in_features = 500, out_features = 10)
#
#     # Definirea filtrelor si modul de conectare intre layere
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, 2, 2)
#         print(x.shape)
#         x = x.view(-1, 4 * 4 * 100)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

###### Functia de antrenare data a retelei neurale
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

###### Functia de testare a retelie neurale
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


###### Rezolvare punctul 9 ###### ! Practic nu ar trebui sa il folosesc
def antrenamentNou(imgDB, epoch, criterion, optimizer):

    running_loss = 0.0

    for i, data in enumerate(imgDB):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = Net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Afisare statistici
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        print('Finished Training')

        # Salvarea retelei neurale
        # torch.save(Net.state_dict(), PATH)


###### Rezolvare punctul 9 #######

# Cred ca as putea sa fac asa:
# - mai fac un script nou de antrenament
# - initializez 3 variabile cu imagini random din baza de date de validare
# - fac un for loop prin care sa trec fiecare imagine si sa o afisez impreuna cu label-ul
# - totul va fi trimis cu un iterator






def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

def imageTrainShow(database, classes, batch_size):
    for i, data, in enumerate(database):
        dataiter = iter(database)
        # get some random training images
        images, labels = dataiter.next()
        # show images
        imshow(torchvision.utils.make_grid(images))
        print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))




def main():
    # Training settings ? De ce a folosit asa sa transfere argumentele?
    # ?Cum functioneaza parser-ul? De ce nu au fost stocate in niste variabile ci au fost folosite argumente?

    ## Argumente folosite pentru algoritmii de antrenare si testare date
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

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    ## Baza de date de antrenament
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                        batch_size=args.batch_size, shuffle=True, **kwargs)

    train_loader_img = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=3, shuffle=True, **kwargs)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    ## Baza de date de validare
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Afisarea a 3 imagini de antrenament
    # imageTrainShow()

    # Definire functie loss and optimizer pentru algoritmul nou de antrenare
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)

    model = Net().to(device) # transferul retelei catre placa video sau procesor
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # for epoch in range(1, args.epochs + 1):
    #    train(args, model, device, train_loader, optimizer, epoch)
    #    test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()

    #network = Net()
    #summary(network)
# epoch cred ca semnifica generatia retelei neurale
