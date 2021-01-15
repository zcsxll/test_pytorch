import torch
import random
import time

def task(rank, cfgs):
    t = random.randint(0, 5)
    #print(time)
    time.sleep(t)
    print(rank, t, cfgs)

def run():
    args = {'k1':1234, 'k2':{'kk1':'asdf', 'kk2':[1, 2, 3, 4]}}
    '''
    args是个元组，这里元组只有一个元素，因此上边的task除去rank，只有一个参数
    '''
    torch.multiprocessing.spawn(task, args=(args, ), nprocs=10)
    print("done")


if __name__ == '__main__':
    run()
