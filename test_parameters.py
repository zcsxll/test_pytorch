import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(5, 8)
        self.fc2 = torch.nn.Linear(8, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

@torch.no_grad()
def diff(model1, model2):
    change = 0
    for p1, p2 in zip(model1, model2.parameters()):
        # print(p1, p2)
        diff = torch.sum(torch.abs(p1-p2))
        change += diff.detach().cpu().numpy()
    return change

@torch.no_grad()
def pclone(model):
    ret = []
    for p in model.parameters():
        ret.append(p.clone())
    return ret

if __name__ == '__main__':
    torch.manual_seed(123)
    model1 = Model()
    model2 = Model()
    m = pclone(model1)
    change = diff(m, model2)
    print(change)
