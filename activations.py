import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return max(0.0, x)

def softmax(x):
    return np.exp(x)/sum(np.exp(x))