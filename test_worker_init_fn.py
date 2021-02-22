import os, sys
import time
import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = [i for i in range(200)]
        print(111)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(self.worker_id, id(self.data))
        tensor = torch.FloatTensor([self.data[idx]])
        time.sleep(1)
        return tensor

    def worker_init_fn(self, worker_id):
        print(worker_id)
        self.worker_id = worker_id
        #self.data = [i for i in range(200)]

if __name__ == '__main__':
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=True,
            batch_size=20,
            num_workers=4,
            worker_init_fn=dataset.worker_init_fn)

    for idx, d in enumerate(dataloader):
        print(idx, d.shape)
