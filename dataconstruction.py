import numpy as np
import random as r
import torch

def makeAdd(start, end, num):
    ## generate num of examples
    x = torch.empty((num, 2))
    y = torch.empty((num, 1))


    for i in range(num):
        x[i][0] = r.randint(start,end)
        x[i][1] = r.randint(start,end)

        y[i] = x[i][0] + x[i][1]

    return x,y

# x,y = makeAdd(0,100,1000)
# print(x)
# print(y)