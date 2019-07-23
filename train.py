import torch
from torch import nn
import numpy as np
import dataconstruction as dc
import matplotlib.pyplot as plt
import time

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def trainTester(model,x,y):
    sumerror=0
    for thing in range(len(x)):
        sumerror += abs(model(x[thing])-y[thing])
    return sumerror/len(x)

def testTester(model,start,end,num):
    x, y = dc.makeAdd(0, 100, 1000)


def main():
    train_x,train_y = dc.makeAdd(0,100,1000)


    model = torch.nn.Sequential(
    # torch.nn.Linear(2, 1),


    # torch.nn.Linear(2, 1),
    # torch.nn.Linear(1, 1)


    torch.nn.Linear(2, 10),
    # torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')
    # learning_rate = 1e-4
    learning_rate = 1e-5
    for t in range(10):
        print(t)
        for ex in range(len(train_x)):

            y_pred = model(train_x[ex])
            loss = loss_fn(y_pred, train_y[ex])
            # print(t, loss.item())
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
    return model,train_x,train_y






if __name__ == "__main__":
    model,testx,testy = main()

    sumerror = trainTester(model,testx,testy)


    def add(x,y):
        return model(torch.tensor([float(x),float(y)]))

    h = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            h[name[2]+name[0]] = param.data.numpy()

    first = np.dot(h['w1'],h['b0'])
    sec = np.dot(h['w1'],h['w0'])[0]
    third = h['b1']

    combined = first+third #Constant term
    mult1 = sec[0] #multiplier on first input
    mult2 = sec[1] #multiplier on second input


    def fadd(x,y):
        return combined+mult1*x+mult2*y



    # print(list(model.named_parameters())[1][0])
    # layer0bias = list(model.named_parameters())[1][1].detach().numpy()
    #
    # print(list(model.named_parameters())[2][0])
    # layer1weight = list(model.named_parameters())[2][1].detach().numpy()
    #
    # layer2weight = list(model.named_parameters())[3][1].detach().numpy()
    #
    # print("add(0,0) should be ")
    # print(np.sum(np.multiply(layer0bias, layer1weight))+layer2weight)
