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

def main():



if __name__ == '__main__':
    main()