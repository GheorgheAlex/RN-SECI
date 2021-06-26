from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import numpy as np

from RN_antrenare import Net

def imshow(img, title):
    mean = 0.1307
    standard = 0.3081
    # img = img / 2 + 0.5  # unnormalize
    img = (img * standard) + mean  # unnormalize
    npimg = img.numpy()
    plt.title(title)
    plt.imshow((np.transpose(npimg, (1, 2, 0))))
    plt.show()

# PUNCT 9
def imagineValidare(database, title, batch_size):
    databaseIter = iter(database)
    images, labels = databaseIter.next()
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    randomNumber = random.randint(0, batch_size)
    imageToShow = images[randomNumber]
    imshow(imageToShow, title)
    print(classes[labels[randomNumber]])

# PUNCT 10
def imagineValidare_Testare(database, title, batch_size, reteaNeurala):
    databaseIter = iter(database)
    images, labels = databaseIter.next()
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    randomNumber = random.randint(0, batch_size)
    imagine_Validare = images[randomNumber]
    imshow(imagine_Validare, title)
    print(classes[labels[randomNumber]])

    reteaNeurala.eval()
    output = reteaNeurala(imagine_Validare)
    pred = torch.argmax(output, 1)
    print(pred)

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        # o epoca se termina atunci cand trecem o data prin toata multimea de antrenare
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        # log-interval-la ce interval se face debug
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")  # Incarcarea retelei neurale pe placa video sau pe cpu

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    test_loader = torch.utils.data.DataLoader(  # clasa de date din pytorch care ne ajuta sa facem procesari de date
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Incarcarea retelei neurale gata antrenate
    modelAntrenat = Net
    modelAntrenat.load_state_dict(torch.load("mnist_cnn.pt"))
    print("Am incarcat modelul gata antrenat in variabila modelAntrenat")

    modelAntrenat = Net().to(device)



    # Afisare imagini din baza de date de validare (PUNCT 9)
    imagineValidare(test_loader, "test", args.test_batch_size)
    imagineValidare(test_loader, "test", args.test_batch_size)
    imagineValidare(test_loader, "test", args.test_batch_size)

    # Incarcare imagini in modelul antrenat (PUNCT 10 - nu merge)
    # imagineValidare_Testare(test_loader, "Imagine validare", args.test_batch_size, modelAntrenat)


    # for epoch in range(1, args.epochs + 1):  # pt fiecare epoca se declara train si test
    #     test(args, modelAntrenat, device, test_loader)

if __name__ == '__main__':
    main()