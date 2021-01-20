import torch
import time
from tqdm import tqdm

class Test:
    def __init__(self, num_workers, pbar=None):
        self.num_workers = num_workers
        self.nums = [i for i in range(20)] 

    def __iter__(self):
        with torch.multiprocessing.Pool(self.num_workers) as pool:
            for m in pool.map(self.cal, self.nums):
                if pbar is not None:
                    pbar.update()
                yield m

    def __len__(self):
        time.usleep(1000000)
        return len(self.nums)

    def cal(self, n):
        return n + 1000

if __name__ == '__main__':
    pbar = tqdm(total=20,
            bar_format='{l_bar}{bar}{r_bar}',
            dynamic_ncols=True)

    test = Test(4, pbar)
    for i in test:
        print(i)
    pbar.close()
