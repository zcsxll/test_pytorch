import torch

torch.manual_seed(123)

t = torch.nn.Linear(2, 5)
print(t.state_dict())
