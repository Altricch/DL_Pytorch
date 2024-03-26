import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

# Retrieve first example
# print(x[0].shape)
# print(x[:,0].shape)
# print(x[2, 0:10])
# x[0,0] = 100


# Fancy indexing
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])

x = torch.rand((3,5))
# print(x)
rows = torch.tensor([1,0])
col = torch.tensor([4,0])
# print(x[rows, col])

# More advanced indexing
x = torch.arange(10)
print(x)
print(x[(x<2) | (x > 8)])
# Modulo operator
print(x[x.remainder(2)==0])

# Useful operations
print(torch.where(x>5, x, x*2))

print(torch.tensor([1,1,2,2,2,3,4,4,4]).unique())
print(x.ndimension())
# number of elements in x
print(x.numel())

