import torch
a = torch.tensor([1.0], requires_grad=True)

b = 2 * a

# with torch.no_grad():
#     c = 2 * b
c = 2 * b

print("123")