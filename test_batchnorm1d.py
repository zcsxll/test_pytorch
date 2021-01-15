import torch
import zcs_util as zu

'''
if training == True: #即调用bn.train()
    计算batch的mean和var

    if track_running_stats == True: #默认的
        根据momentum和eps以及mean和var更新全局的running_mean和running_var
    elif track_running_stats == False:
        没有running_mean和running_var变量

    把batch减去mean，然后除以sqrt(var)，即归一化到均值0，方差1

    if affine == True: #默认的
        使用weight和bias进行仿射变换，即乘以weight后，加bias
    elif affine == False:
        没有weight和bias变量，什么都不做

    #如果两个都是False，则num_batches_tracked也没有了，BN层就没有任何参数了

elif training == False: #即调用bn.eval()，无论如何都不会再更新running_mean和running_var
    if track_running_stats == True: #默认的
        把batch减去running_mean，然后除以sqrt(running_var)
    elif track_running_stats == False:
        没有变量running_mean和running_var，把batch减去batch自己的mean，然后除以sqrt(batch自己的var)，即归一化到均值0，方差1

    if affine == True: #默认的
        使用weight和bias进行仿射变换，即乘以weight后，加bias
    elif affine == False:
        没有weight和bias变量

    #如果两个都是False，则num_batches_tracked也没有了，BN层就没有任何参数了
'''

def test():
    bn = torch.nn.BatchNorm1d(
        num_features=4,
        eps=100,
        momentum=0.4,
        affine=True,
        track_running_stats=True) #出了第一个，其余都是默认参数

    tensor = torch.FloatTensor([[1, 2, 4, 1],
                                [6, 3, 2, 4],
                                [2, 4, 6, 1]])
    print(tensor.mean(dim=0), tensor.var(dim=0, unbiased=False))
    tensor = torch.FloatTensor([1, 2, 4, 2]).view(-1, 4)
    print(tensor)

    # bn_weight= torch.FloatTensor([1, 1, 1, 1])
    # zu.set_state_dict(bn, 'weight', bn_weight)
    # bn_bias = torch.FloatTensor([0, 0, 1, 10])
    # zu.set_state_dict(bn, 'bias', bn_bias)

    # bn_mean = torch.FloatTensor([1, 1, 0, 0])
    # zu.set_state_dict(bn, 'running_mean', bn_mean)
    # bn_var = torch.FloatTensor([1, 1, 1, 0])
    # zu.set_state_dict(bn, 'running_var', bn_var)

    zu.print_state_dict(bn)
    bn.eval()
    output = bn(tensor)
    zu.print_state_dict(bn)
    print(output)

if __name__ == '__main__':
    test()