import mmap
import torch
import numpy as np
import os

class TensorLoader:
    def __init__(self, m: int, n: int):
        self.files = []
        self.m = m
        self.n = n

    def load(self, path: str):
        if os.path.getsize(path) != self.m * self.n * 4:
            raise Exception(f'file size is expected to be {self.m}*{self.n}*4')
        f = open(path, 'r+b')
        self.files.append(f)
        buffer = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return torch.Tensor(np.frombuffer(buffer, dtype=np.float32, count=self.m*self.n).reshape((self.m, self.n)))

    def close(self):
        for f in self.files:
            f.close()
