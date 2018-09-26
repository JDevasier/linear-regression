import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

def main():
    data = pd.read_csv('student.csv')
    #print(data.shape)

    axis_list = []

    for val in data.columns.values:
        axis_list.append(val)

    M = len(data[axis_list[0]])
    
    X = [np.ones(M)]

    for axis in axis_list:
        X = np.vstack((X, data[axis]))

    
    #b0, b1 = getBeta(X, Y)
    #x = np.linspace(np.min(X) * 0.85, np.max(X) * 1.2, 1000)

    #plt.scatter(data['Head Size(cm^3)'].values, data['Brain Weight(grams)'].values, color='#00FF00')
    #plt.plot(x, b0 + b1 * x, color='#FF0000')
    #plt.show()
def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

def getBeta(axis_list):
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    m = len(X)

    numer = 0
    denom = 0
    for i in range(m):
        numer += (X[i] - mean_x) * (Y[i] - mean_y)
        denom += (X[i] - mean_x) ** 2
    b1 = numer / denom
    b0 = mean_y - (b1 * mean_x)
    return (b0, b1)

def createPlot(x, y):
    plt.plot(x, y)
    plt.show()

main()