import torch
import numpy as np

import zcs_util as zu

def test():
    data = np.arange(4*3*5*2, dtype=np.float32).reshape(4, 3, 5, 2)
    tensor = torch.from_numpy(data)
    # print(tensor)

    bn = torch.nn.BatchNorm2d(num_features=3)
    bn.train()

    out = bn(tensor)
    # print(bn)
    zu.print_state_dict(bn)

if __name__ == '__main__':
    test()
