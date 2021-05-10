import torch
import torch.nn as nn
import torch.nn.functional as F

### Definirea retelei neurale

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 canal de intrare pt imagine, 6 canale de iesire si 5x5 square convolution
        #kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): # Doar asta trebuie definita, functia backward e definita automat de autograd
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)
print()

## Metode de a afla cati parametrii learnable are un model creat si afisarea lor
#params = list(net.parameters())
#print(len(params))
#print(params[3].size())  # conv1's .weight

# Aici am introdus un input random de 32x32
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

print()
### Loss Function
# A loss function takes the (output, target)
# pair of inputs, and computes a value that
# estimates how far away the output is from the target.

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
lossBCK = criterion(output, target)
print(loss)
print(loss.backward())

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

### Backprop

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

### Update the weights

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)



