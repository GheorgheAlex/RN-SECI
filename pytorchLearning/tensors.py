import torch
import numpy as np

##### Metode de creare a tensorilor

## Direct din date
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

## Cu un array Numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

## Din alt tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
#print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
#print(f"Random Tensor: \n {x_rand} \n")

## Cu valori random si valori constante

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
#print(f"Random Tensor: \n {rand_tensor} \n")
#print(f"Ones Tensor: \n {ones_tensor} \n")
#print(f"Zeros Tensor: \n {zeros_tensor}")


##### Atributele tensorilor

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


##### Operatii cu tensori

### Mutarea pe alt dispozitiv (de ex pe placa video)

if torch.cuda.is_available():
  tensor = tensor.to('cuda')

### Indexing and slicing

tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

### Joining tensors

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

###  Multiplying tensors

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
# Si multiplicare matriceala
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

### Conversie din array la tensor
n = np.ones(5)
t = torch.from_numpy(n)


