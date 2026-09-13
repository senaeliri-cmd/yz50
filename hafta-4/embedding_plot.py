import torch
import matplotlib.pyplot as plt
from forward_pass import itos

checkpoint = torch.load("model.pt")
C = checkpoint["C"]
plt.scatter(C[:,0].data, C[:,1].data)

for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i])

plt.show()