from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import random

class Net(nn.Module):

   def __init__(self):
       super(Net, self).__init__()
       self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = 5, stride = 1)
       self.conv1_batch = nn.BatchNorm2d(20)
       self.conv2 = nn.Conv2d(in_channels = 20, out_channels = 50, kernel_size = 3, stride = 1)
       self.conv2_batch = nn.BatchNorm2d(50)
       self.conv3 = nn.Conv2d(in_channels= 50, out_channels = 100, kernel_size = 3, stride=1)
       self.conv3_batch = nn.BatchNorm2d(100)
       self.dropout = nn.Dropout(0.25)
       self.fc1 = nn.Linear(in_features = 1 * 1 * 100, out_features = 500)
       self.fc2 = nn.Linear(in_features = 500, out_features = 10)

   def forward(self, x):
       x = F.relu(self.conv1(x))
       x = F.max_pool2d(x, 2, 2)
       x = self.conv1_batch(x)
       x = F.relu(self.conv2(x))
       x = F.max_pool2d(x, 2, 2 , 1)
       x = self.conv2_batch(x)
       x = F.relu(self.conv3(x))
       x = F.max_pool2d(x, 2, 2, 1)
       x = F.max_pool2d(x, 2, 2)
       x = self.conv3_batch(x)
       x = x.view(-1, 1 * 1 * 100)
       x = F.relu(self.fc1(x))
       x = self.dropout(x)
       x = self.fc2(x)

       return F.log_softmax(x, dim=1)

#-------- Functia de afisare a unei imagini --------#
def imshow(img, title):
        mean = 0.1307
        standard = 0.3081
        # img = img / 2 + 0.5  # unnormalize
        img = (img * standard) + mean  # unnormalize
        npimg = img.numpy()
        plt.title(title)
        plt.imshow((np.transpose(npimg, (1, 2, 0))))
        plt.show()
#---------------------------------------------------#

#-------- Algoritm de afisare al imaginii din baza de date de validare --------#
def imageShowLabels(image,labels, labelImageNumber, title):
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    print('Validation images labels: ')
    print(" ".join('%1s' % classes[labels[labelImageNumber]]))
    imshow(torchvision.utils.make_grid(image), title)

#-----------------------------------------------------------------------------#

#-------- Functia de validare a retelei neurale folosind toata baza de date de validare --------#
def testWholeBatch(args, model, device, test_loader):
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
#-------------------------------------------------------------------------------------------------#

#-------- Functia de validare a retelei neurale --------#
def test(args, model, device, data, target):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        print(pred)
        print(correct)
        print(test_loss)

    # test_loss /= len(data.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
#------------------------------------------------------#

# Punctul 10
def labelTest(model, testImage):
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    model.eval()
    output = model(testImage)
    # pred = torch.argmax(output, 1)
    _, predicted = torch.max(output, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(1)))

def main():

    # Argumente folosite pentru algoritmii de antrenare si testare date
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch-size-img', type=int, default=3, metavar='N',
                        help='input training images to show to user (default: 3)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
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
                        help='random seed (default: 1)')  # ? Ce reprezinta seed-ul?
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")  # Incarcarea retelei neurale pe placa video sau pe cpu

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Incarcare baza de date de validare
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Se parcurge si itereaza baza de date de validare
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # Generez 3 valori random pentru a alege imaginile
    randomNumberImage1 = random.randint(0, args.test_batch_size)
    randomNumberImage2 = random.randint(0, args.test_batch_size)
    randomNumberImage3 = random.randint(0, args.test_batch_size)

    # Aleg imaginile si le pun in variabile pentru a le testa alaturi de label-uri
    image1 = images[randomNumberImage1]
    label1 = labels[randomNumberImage1]

    image2 = images[randomNumberImage2]
    label2 = labels[randomNumberImage2]

    image3 = images[randomNumberImage3]
    label3 = labels[randomNumberImage3]

    # ----------------- Rezolvare punct 9 ------------------#
    # Afisarea a 3 imagini din baza de date de validare
    imageShowLabels(image1, labels, randomNumberImage1, "Image 1")
    imageShowLabels(image2, labels, randomNumberImage2, "Image 2")
    imageShowLabels(image3, labels, randomNumberImage3, "Image 3")
    # ----------------------------------------------------#

    # ----------------- Rezolvare punct 10 ------------------#
    # Se instantiaza modelul salvat intr-o alta variabila
    trainedModel = Net()
    trainedModel.load_state_dict(torch.load("mnist_cnn.pt"))
    print("Am incarcat modelul salvat in variabila trainedModel")
    trainedModel = Net().to(device)

    # Conversie imagini inainte de a le incarca in retea
    image1Net = image1.unsqueeze(1)
    label1Net = label1.unsqueeze(0)
    image2Net = image2.unsqueeze(1)
    label2Net = label2.unsqueeze(0)
    image3Net = image3.unsqueeze(1)
    label3Net = label3.unsqueeze(0)

    # Se muta imagininile pe acelasi device pe care e reteaua
    image1Net, label1Net = image1Net.to(device), label1Net.to(device)
    image2Net, label2Net = image2Net.to(device), label2Net.to(device)
    image3Net, label3Net = image3Net.to(device), label3Net.to(device)


    # Incarcare imagini in reteaua neurala
    labelTest(trainedModel, image1Net)
    labelTest(trainedModel, image2Net)
    labelTest(trainedModel, image3Net)


    # Validarea retelei folosind toata baza de date de validare
    testWholeBatch(args, trainedModel, device, test_loader)

if __name__ == '__main__':
    main()