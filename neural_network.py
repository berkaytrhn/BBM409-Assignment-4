import numpy as np
from activations import *
from losses import *
from utils import *
import sys
from sklearn.metrics import accuracy_score


class NeuralNetwork:

    def __init__(self, X, y):
        self.input = X
        self.y = y
        self.output_size = len(np.unique(y))

        self.num_examples = len(y)
        self.weights1 = np.random.rand(self.input.shape[1], self.output_size) # attributes_of_x, 3
        #self.weights2 = np.random.rand(4,3)

        
        

    def fit(self, epochs, learning_rate):
        self.learning_rate = learning_rate
        history = dict(
            {
                "accuracy": [],
                "loss": []
            }
        )

        for i in range(epochs):
            self.feed_forward(self.input)

            _loss, _ = nll_loss(self.layer2, self.y)
            predicted = np.argmax(self.layer2, axis=1)
            _accuracy = accuracy_score(self.y, predicted)

            history["accuracy"].append(_accuracy)
            history["loss"].append(_loss)
            sys.stdout.write(f"\rEpochs: {i+1} -> Accuracy: {round(_accuracy, 5)} - Loss: {round(_loss, 5)}")
            sys.stdout.flush()
            
            self.back_propagate()
        
        return history

    def predict(self, X):
        self.feed_forward(X)

        predictions = np.argmax(self.layer2, axis=1)

        return predictions

        

    def feed_forward(self, data):
        relu_vec = np.vectorize(relu)
        softmax_vec = np.vectorize(softmax)

        #self.layer1 = relu_vec(np.dot(self.input, self.weights1))
        self.layer2 = np.apply_along_axis(softmax, 1, np.dot(data, self.weights1))
        #print(self.layer2)


    def back_propagate(self):
        #print("back propagation")
        dscores = self.layer2
        #print(dscores, "\n**")
        dscores[range(len(self.y)), self.y] -= 1
        #print(dscores, "\n**")
        dscores /= self.num_examples
        #print(dscores, "\n**")
        #print("transpose X")
        #print(self.input.T)
        #print("**dot product")
        #print(np.dot(self.input.T, dscores))
        #print("weight update")
        #print(self.weights2)
        self.weights1 += -self.learning_rate*np.dot(self.input.T, dscores)
        #print("after")
        #print(self.weights2)

    def save_weights(self):
        np.save("weights.npy", self.weights1)

    def load_weights(self, file_name):
        # .npy file
        self.weights1 = np.load(file_name)


    