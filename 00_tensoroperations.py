import torch 

##################################
######  TENSOR INITIALIZATION ####
##################################

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32, device=device,
                         requires_grad=True)
# print(my_tensor)
# print(my_tensor.dtype)
# print(my_tensor.device)
# print(my_tensor.shape)
# print(my_tensor.requires_grad)


# Other common initialization
x = torch.empty(size=(3,3))
x = torch.zeros(size = (3,3))
x = torch.rand(size = (3,3))
x = torch.ones(size = (3,3))
x = torch.eye(3,3)
# Like python range
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)

# Like arange with uniform distribution
x = torch.empty(size=(1,5)).normal_(0,1)
x = torch.empty(size=(1,5)).uniform_(0,1)

# Like eye, but we can perserve values on the diagonal
x = torch.diag(torch.ones(3))
# print(x)


################### initialize and convert tensort to other types
tensor = torch.arange(4)
print(tensor)
# converts it into boolean values
print(tensor.bool())
# converts to int16
print(tensor.short())
# converts to int64
print(tensor.long())\
# convert to float16
print(tensor.half())

import numpy as np

# Conversion from NP to tensor and back
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()

