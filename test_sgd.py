import torch
# from torch import Optimizer
# from .optimizer import Optimizer, required

class MySGD(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        # if lr is not required and lr < 0.0:
        #     raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MySGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        print('1111111', buf)
                    else:
                        buf = param_state['momentum_buffer']
                        print('2222222', buf)
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss

def function_y(w):
    # x = torch.tensor([[5.0,3.0], [-3.0,-4.0]])
    # y = torch.mm(x, w) + torch.tensor([[-1], [1]])
    # # print(y, y.pow(2))
    # y = torch.sum(y.pow(2))
    # # print(y)
    y = w[0] ** 2 + torch.log(w[1])
    return y

def go():
    # weight = torch.randn((2, 1), requires_grad=True)
    weight = torch.FloatTensor([7, 5])
    weight.requires_grad=True

    optim = MySGD([weight], lr=0.1, momentum=0.5)
    # optim = torch.optim.SGD([weight], lr=0.1)

    for step in range(5):
        output = function_y(weight)

        optim.zero_grad()
        output.backward()
        print(weight.grad)
        optim.step()

        # print(output.grad)
        print(weight)
        # grad = torch.autograd.grad(output, weight)
        # print(grad)

if __name__ == '__main__':
    go()
    # a = torch.FloatTensor([5])
    # print(a)
    # a.mul_(4).add_(100)
    # print(a)