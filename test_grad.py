import torch

def run():
    a = torch.randn((3, 4), requires_grad=True)
    print(a.grad_fn)

    b = a * 2
    print(type(b.grad_fn))

run()
