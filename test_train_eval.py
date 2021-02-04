import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc = torch.nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        x = self.fc(x)
        print(self.training)
        return x

tensor = torch.FloatTensor(2, 10)
print(tensor.shape)

model = Model()
model.train()
out = model(tensor)
model.eval()
out = model(tensor)

