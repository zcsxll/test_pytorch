import torch

class Model:
    def __init__(self):
        self.a = 123
        self.b = '3345'

        print(self.__dict__)

    def froward(self, x):
        return x + 1

model = Model()
print(model.__dict__)
