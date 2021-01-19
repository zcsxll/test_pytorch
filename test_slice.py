import numpy as np
import torch

def test():
    a = np.arange(2*3*4).reshape(2, 3, 4)
    print(a)
    print()
    print()

    b = a[:, None, :]
    print(b.shape)

if __name__ == '__main__':
    test()