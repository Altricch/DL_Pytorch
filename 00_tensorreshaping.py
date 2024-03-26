import torch

############################
##### Tensor Reshaping #####
############################

x = torch.arange(9)

# View and shape very similar, simply put
# View acts on contiguious tensors in memory
# Use reshape generally, for performance use view 
x_3x3 = x.view(3,3)
x_3x3 = x.reshape(3,3)

x = torch.ones((2,5))
y = torch.rand((2,5))

# z = torch.cat((x,y), dim=1)
# print(z)

