import torch
import zcs_util as zu

def test():
    bn = torch.nn.LayerNorm(
        normalized_shape=4,
        eps=1e-05,
        elementwise_affine=False) #除了第一个，其余都是默认参数

    tensor = torch.FloatTensor([[1, 2, 4, 1],
                                [6, 3, 2, 4],
                                [2, 4, 6, 1]])
    print(tensor.mean(dim=-1), tensor.var(dim=-1, unbiased=False))
    # tensor = torch.FloatTensor([1, 2, 4, 2]).view(-1, 4)
    print(tensor)

    # bn_weight= torch.FloatTensor([1, 1, 1, 10])
    # zu.set_state_dict(bn, 'weight', bn_weight)
    # bn_bias = torch.FloatTensor([0, 0, 1, 10])
    # zu.set_state_dict(bn, 'bias', bn_bias)

    # bn_mean = torch.FloatTensor([1, 1, 0, 0])
    # zu.set_state_dict(bn, 'running_mean', bn_mean)
    # bn_var = torch.FloatTensor([1, 1, 1, 0])
    # zu.set_state_dict(bn, 'running_var', bn_var)

    zu.print_state_dict(bn)
    #bn.eval()
    output = bn(tensor)
    zu.print_state_dict(bn)
    print(output)

def test2():
    bn = torch.nn.LayerNorm(
        normalized_shape=torch.Size([3, 4]),
        eps=1e-05,
        elementwise_affine=True) #除了第一个，其余都是默认参数

    # tensor = torch.FloatTensor([[1, 2, 4, 1],
    #                             [6, 3, 2, 4],
    #                             [2, 4, 6, 1]])
                                
    tensor = torch.FloatTensor([[[1, 2, 4, 1],
                                [6, 3, 2, 4],
                                [2, 4, 6, 1]],
                                [[1, 6, 4, 1],
                                [6, 4, 2, 4],
                                [3, 4, 6, 6]]])
    # tensor = torch.rand(10, 100, 3, 4)
    print(tensor[0].mean(),tensor[1].mean())
    print(tensor[0].var(unbiased=False), tensor[1].var(unbiased=False))
    # tensor = torch.FloatTensor([1, 2, 4, 2]).view(-1, 4)
    print(tensor)

    # bn_weight= torch.FloatTensor([1, 1, 1, 10])
    # zu.set_state_dict(bn, 'weight', bn_weight)
    bn_bias = torch.FloatTensor([[0, 0, 1, 10],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]])
    # zu.set_state_dict(bn, 'bias', bn_bias)

    # bn_mean = torch.FloatTensor([1, 1, 0, 0])
    # zu.set_state_dict(bn, 'running_mean', bn_mean)
    # bn_var = torch.FloatTensor([1, 1, 1, 0])
    # zu.set_state_dict(bn, 'running_var', bn_var)

    zu.print_state_dict(bn)
    bn.train()
    output = bn(tensor)
    # zu.print_state_dict(bn)
    print(output)

if __name__ == '__main__':
    test()
