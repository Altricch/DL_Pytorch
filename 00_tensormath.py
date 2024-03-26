import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

#####################
### Tensor Math #####
#####################


x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# Addition
z1 = torch.empty(3)
torch.add(x,y,out=z1)
# print(z1)
z1 = x + y
# print(z1)

# Division (element wise division if of equal shape)
z = torch.true_divide(x,y)

# inplace operations - any operation with "_" ad the end is an inplace operation
t = torch.zeros(3)
t.add_(x)
t += x # t = t + x is not in place

# Exponentiation
z = x.pow(2) # elementwise power of 2
z = x ** 2 # same as above

# Simple comparison
z = x > 0 # element wise comparison

# Matrix multiplication
x = torch.rand(2,5)
y = torch.rand(5,3)
x3 = x.mm(y)

# Matrix exponentiation
matrix_exponent = torch.rand(5,5)
# Three times multiplied by itself
matrix_exponent.matrix_power(3)

# Element wise mult
x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])
z = x * y
# print(z)

# Dot product
z = torch.dot(x, y)
print(z)

# Batch Matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(tensor1, tensor2) # (Batch, n, p)

# Examples of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))
# Vector is substracted by each row in the matrix (the expansion is called Broadcasting)
z = x1 - x2
# Same here, broadcasting elementwise exponentiation
z = x1 ** x2

# Other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
# same as max without values
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
# mean required float
mean_x = torch.mean(x.float(), dim=0)
# checks for equality, boolean output
z = torch.eq(x,y)
# sorting
sorted_y, indices = torch.sort(y, dim=0, descending=False)
# Check all elements and set it to 0 if any smaller than 0 (e.g. clamping)
z = torch.clamp(x, min=0)

x = torch.tensor([0,1,1,1,1]).bool()
z = torch.any(x)
print(z)
z = torch.all(x)
print(z)