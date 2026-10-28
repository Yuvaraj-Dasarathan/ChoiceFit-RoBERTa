import torch

layer = torch.nn.Linear(3,5)

print("weight:", layer.weight)
print("bias:", layer.bias)