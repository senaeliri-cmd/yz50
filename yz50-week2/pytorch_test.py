import torch

a1 = torch.tensor([0.89], dtype = torch.double, requires_grad = True)
a2 = torch.tensor([2.12], dtype = torch.double, requires_grad= True)

w1 = torch.tensor([2.34], dtype = torch.double, requires_grad= True)
w2 = torch.tensor([-1.12], dtype = torch.double, requires_grad = True)

b = torch.tensor([1.02], dtype= torch.double, requires_grad= True)

n = a1*w1 + a2*w2 + b
o = torch.tanh(n)
print(o.data.item())
o.backward()

print(f"a1 data item {a1.grad.item()}")
print(f"a2 data item {a2.grad.item()}")

print(f"w1 data item {w1.grad.item()}")
print(f"w2 data item {w2.grad.item()}")

#0.621962893112366
#a1 data item 1.4347994534436317
#a2 data item -0.6867416187422511
#w1 data item 0.5457143220362531
#w2 data item 1.2999037783335468
