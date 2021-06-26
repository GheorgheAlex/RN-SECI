from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self): #descriem layerele pe care le avem de folosit
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 3, 1,padding=1) #dupa conv, size-ul este de 20 kernele
        self.conv2 = nn.Conv2d(30, 50, 3, 1)
        #conv1 ->conv2 -add-daca au acelasi nr de canale(/concat)- conv3 (model rezidual ResNet-8a
        self.conv3 = nn.Conv2d(50,70, 3, 1)
        self.fc1 = nn.Linear(6*6*70, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x): #descrie layerele pe care trebuie sa le folosim
        x = F.relu(self.conv1(x)) #functie de activare pt toate datele de la intrare->20,26,26
        x = F.max_pool2d(x, 2, 2) #primul 2=kernel size=2(matrice 2/2), al 2-lea=strike(se deplaseaza matricea cu 2 poz) 20,13/13
       #->20,13,13
        x = F.relu(self.conv2(x))
        #->70,11,11
        x = F.max_pool2d(x, 2, 2)
        #->5,4,4
        x = x.view(-1, 4*4*50) #view face resize,vectorizarea valorilor(la iesire un vector de 4*4*50, si prima iesire e ce ramane
        x = F.relu(self.fc1(x)) #500el
        x = self.fc2(x)#10 el
        return F.log_softmax(x, dim=1) #din cauza functiei de nll_loss(negatively likelihood loss) se pune log_softmax
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): #batch-se face media si variatia datelor pt a putea fi normalizate
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data) #conversie din tensor in array
        #pycharm tensor to ndarray(tensorul din data trebe sa-l facem 28,28,1)
        #din data [64,2,28,28] luam de ex data[0] -?[1,28,28] -> [28,28,1] ->numpy  ndarray[28,28,1] ->im show sasu imwrite
        #                                 data[1]
        #                                 data[2]
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

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
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', #o epoca se termina atunci cand trecem o data prin toata multimea de antrenare
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', #lr-learning rate
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', #log-interval-la ce interval se face debug
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False, #daca se face sau nu save la model
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
                           transforms.Normalize((0.1307,), (0.3081,)) #preprocesarea-normalizare
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(  #clasa de date din pytorch care ne ajuta sa facem procesari de date
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device) #se declara un model de tip net si se pune pe device
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) #se declara un optimizator

    for epoch in range(1, args.epochs + 1): #pt fiecare epoca se declara train si test
        train(args, model, device, train_loader, optimizer, epoch)
        # test(args, model, device, test_loader)

    if (args.save_model): #dupa fiecare epoca se salveaza modelul
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
    #feature extraction- cu conv-pool-conv
    #pool nu are param care se invata
    #feature_clasification-cu fc
    #Imag-feature_extraction-feature_clasification-decision
    #pt 9 tre sa facem un nou proiect si in loc de train ii dam load-partea de testare o copiem