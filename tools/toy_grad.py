import torch
import torch.nn as nn
class T(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Linear(2, 2)
        self.alpha = nn.Linear(2, 2)
    def forward(self, x):
        return self.l(self.weight)
    
a = torch.ones(2, 2)
model = T()

optm = torch.optim.SGD(model.parameters(), 1e-3)
label = torch.tensor([1,0])
ct=torch.nn.CrossEntropyLoss()

result = model(a)
loss=ct(result[0], label)
opt.zero_grad()
loss.backward()