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


#-------- Algoritm de afisare al imaginii din baza de date de validare --------#
def imageShowLabels(database, title):
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    # for i, data, in enumerate(database):
    dataiter = iter(database)
    images, labels = dataiter.next()
    # show images
    print('Label: '.join('%5s' % classes[labels[i]] for i in range(3))) # de rezolvat formatul la print cas a arat mai bine label-urile

    imshow(torchvision.utils.make_grid(images[0:3]), title)
    # print(' '.join('%5s' % classes[labels[j]] for j in range(3)))

#-----------------------------------------------------------------------------#

#-------- Functia de validare a retelei neurale --------#
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

def testOutput(args, model, device, test_loader):
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
    print(model.eval())
#------------------------------------------------------#


def main():

    # Argumente folosite pentru algoritmii de antrenare si testare date
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
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

    # Incarcare baza de date de validare
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # 3 imagini aleatorii luate din baza de date validare pentru testare
    test_loader_img1 = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True, **kwargs)

    test_loader_img2 = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True, **kwargs)

    test_loader_img3 = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True, **kwargs)

    # ----------------- Rezolvare pct 9 ------------------#
    # Afisarea a 3 imagini din baza de date de validare
    # imageShowLabels(test_loader_img1, "imagine1")
    # imageShowLabels(test_loader_img2, "imagine2")
    # imageShowLabels(test_loader_img3, "imagine3")
    # ----------------------------------------------------#

    # ----------------- Rezolvare pct 10 ------------------#

    # Antrenarea retelei neurale folosind baza de date de antrenare
    # for epoch in range(1, args.epochs + 1):
    #    train(args, model, device, train_loader, optimizer, epoch)

    # Dupa antrenare se salveaza modelul
    # if (args.save_model):
    #     torch.save(model.state_dict(), "mnist_cnn.pt")
    #     print("Am salvat reteaua neurala in memoria PC-ului")

    # Se instantiaza modelul salvat intr-o alta variabila
    newModel = Net()
    newModel.load_state_dict(torch.load("mnist_cnn.pt"))
    print("Am incarcat modelul salvat in variabila newModel")
    newModel = Net().to(device)

    # Rulare algoritm de validare folosind reteaua stocata

    # imageShowLabels(test_loader_img1, "imagine1")
    # test(args, newModel, device, test_loader_img1)

    # imageShowLabels(test_loader_img2, "imagine2")
    # test(args, newModel, device, test_loader_img2)

    # imageShowLabels(test_loader_img3, "imagine3")
    # test(args, newModel, device, test_loader_img3)

    # ----------------------------------------------------#



if __name__ == '__main__':
    main()