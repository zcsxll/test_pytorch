import torch

def print_state_dict(module):
    print('==================================')
    for key, val in module.state_dict().items():
        print(key, '\t', val)
    print('==================================')

def set_state_dict(module, key, val):
    state_dict = module.state_dict()
    # print(state_dict.keys())
    assert key in state_dict.keys()
    state_dict[key] = val
    module.load_state_dict(state_dict)
    # return module