import torch

class Test:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.nums = [i for i in range(20)] 

    def __iter__(self):
        with torch.multiprocessing.Pool(self.num_workers) as pool:
            for m in pool.map(self.cal, self.nums):
                yield m

    def cal(self, n):
        return n + 1000

if __name__ == '__main__':
    test = Test(4)
    for i in test:
        print(i)
