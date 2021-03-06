import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import math

def main():
    train_file = "pendigits_training.txt"
    deg = 1 # deg (1 or 2) is the power the vals of the basis fun are raised to
    test_file = "pendigits_test.txt"

    logistic_regression(train_file, deg, test_file)


def logistic_regression(train_file, degree, test_file):
    train_data = read_file(train_file)
    test_data = read_file(test_file)

    train_labels = convertClassLabels(train_data)
    test_labels = convertClassLabels(test_data)

    N = np.shape(train_data)[0]
    D = np.shape(train_data)[1] - 1

    w_old = np.zeros(len(train_data[0][:-1]))

    t = np.asarray(train_labels)
    X = np.asarray(train_data)
    X = X[:,:-1]

    I = np.zeros((N, D))
    y = np.zeros(t.shape)
    R = np.zeros((N, N))

    for i in range(N):
        I[i] = np.asarray(basis_function(X[i], degree))

    while True:        
        
        for i in range(N):
            y[i] = sigmoid(np.dot(w_old, basis_function(X[i], degree)))

        #N = number of training_samples
        #D = number of dimensions in training sample
        #I = NxD array of D basis functions applied to N observations(?)
        #t = 1xN array of the real class of the observation
        #y = 1xN array of the estimated probability of C1 given x
        #R = diag matrix s.t. R_nn = y_n(1-y_n)
        
        for i in range(N):
            R[i][i] = y[i] * (1 - y[i])

        H = np.linalg.pinv(np.matmul(np.matmul(np.transpose(I), R), I))
        w_change = np.matmul(H, np.matmul(np.transpose(I), y-t))
        w_new = w_old - w_change

        #print(w_old, w_change, w_old - w_new <= 0.000000000001)

        #print(w_new, np.sum(w_new-w_old))

        if (w_old - w_new).all() < 0.001:
            break

        w_old = w_new

    for i in range(len(w_old)):
        print("w{:d} = {:.4f}".format(i, w_old[i]))
    
        ## NOW REPEAT THE CALCULATION

    correct = 0
    for i in range(len(test_labels)):
        _y = find_y(test_data[i][:-1], w_old, degree)
        acc = 0
        if round(_y) == test_labels[i]:
            acc = 1
        print("ID={:5d}, predicted={:3d}, probability={:.4f}, true={:3d}, accuracy={:4.2f}".format(i, round(_y), _y, test_labels[i], acc))

        #cutoff for classification accuracy is 0.5 by rounding
        if test_labels[i] == round(_y):
            correct += 1
    print("classification accuracy={:6.4f}".format(correct / len(test_labels)))



    
def find_y(x, w, deg):
    return sigmoid(np.dot(w, basis_function(x, deg)))

def convertClassLabels(data):
    labels = []
    for line in data:
        v = 0
        if line[-1] == 1:
            v = 1
        labels.append(v)
    return labels

def sigmoid(a):
    return 1 / (1 + math.exp(-a))

def basis_function(row, deg):
    new = [1]

    if deg == 1:
        for i in range(1, len(row)): 
            new.append(row[i])

    elif deg == 2:
        for i in range(1, len(row)): 
            #new.append(row[i])
            new.append(math.pow(row[i], deg))
        
    return new

def read_file(fname):
    lines = []
    with open(fname) as f:
        for line in f:
            lines.append(list(map(float, line.split())))

    return lines
