import torch
import torchvision

### Codul ilustreaza metoda si tot ce e necesar pentru a antrena o retea neurala

# Luam un model gata antrenat
model = torchvision.models.resnet18(pretrained = True)

# Generam date random care reprezinta imagini cu 3 canale de culoare cu dimens 64 x 64
# Datele au si labels corespondente
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# Rulam datele prin model - asta se numeste forward pass
prediction = model(data)

# Folosim predictia generata sa calculam eroarea
# Urmatorul pas este sa repropagam eroarea prin retea - se numeste backward pass

loss = (prediction - labels).sum()
loss.backward()

# Incarcam un optimizator ; lr - learning rate
optim = torch.optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9)

# La final, pornim descresterea gradientului

optim.step()

