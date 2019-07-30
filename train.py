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

    # Generate training examples and initialize models
    train_x,train_y = dc.makeAdd(0,100,1000)

    modelA = torch.nn.Sequential(torch.nn.Linear(2,1))
    modelC = torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.Linear(10, 1))



    # Set the learning rate for backprop and decide which loss function to use
    loss_fn = torch.nn.MSELoss(reduction='sum')
    # learning_rate = 1e-4
    learning_rate = 1e-5


    # Backprop
    for t in range(1):
        for ex in range(len(train_x)):

            y_predA = modelA(train_x[ex])
            y_predC = modelC(train_x[ex])
            lossA = loss_fn(y_predA, train_y[ex])
            lossC = loss_fn(y_predC, train_y[ex])
            modelA.zero_grad()
            modelC.zero_grad()
            lossA.backward()
            lossC.backward()

            with torch.no_grad():
                for param in modelA.parameters():
                    param -= learning_rate * param.grad
                for param in modelC.parameters():
                    param -= learning_rate * param.grad

    return modelA, modelC, train_x, train_y






if __name__ == "__main__":
    modelA, modelC, _, _ = main()



    # Score the trained algorithms on a new test set and print the results
    testx, testy = dc.makeAdd(0, 100, 1000)

    print("A average error: " + str(trainTester(modelA,testx,testy)))
    print("C average error: " + str(trainTester(modelC,testx,testy)))


    # Make addition testing easier - MANUAL ONLY!
    # If running loops, use madd below to avoid the time cost of type conversion every operation
    def addA(x,y):
        return modelA(torch.tensor([float(x),float(y)]))
    def addC(x,y):
        return modelC(torch.tensor([float(x),float(y)]))

    # Make comparing runtimes easier
    m = torch.tensor([float(1), float(1)])

    def maddA(mat):
        return modelA(mat)
    def maddC(mat):
        return modelC(mat)




    # Extract learned parameters from models A and C
    paramsA = {}
    for name, param in modelA.named_parameters():
        if param.requires_grad:
            paramsA[name[2]+name[0]] = param.data.numpy()
    paramsC = {}
    for name, param in modelC.named_parameters():
        if param.requires_grad:
            paramsC[name[2]+name[0]] = param.data.numpy()



    # Precalculation of matrix multiplication for model C
    first = np.dot(paramsC['w1'],paramsC['b0'])
    sec = np.dot(paramsC['w1'],paramsC['w0'])[0]
    third = paramsC['b1']

    combined = first+third #Constant term
    mult1 = sec[0] #multiplier on first input
    mult2 = sec[1] #multiplier on second input


    def fadd(x,y):
        return combined+mult1*x+mult2*y
