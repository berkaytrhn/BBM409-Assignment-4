import numpy as np
from sklearn.utils import validation
from activations import *
from losses import *
from utils import *
import sys
from sklearn.metrics import accuracy_score
import os

class NeuralNetwork:

    def __init__(self, X, y):
        self.input = X
        self.y = y
        self.output_size = len(np.unique(y))

        self.num_examples = len(y)
        # 4 -> 8
        # 8 -> 3
        self.weights1 = np.random.rand(self.input.shape[1], 8) # attributes_of_x, 3
        self.weights2 = np.random.rand(8, self.output_size)


    def fit_alternative(self, epochs, learning_rate, validation_set=None):
        self.learning_rate = learning_rate
        for i in range(epochs):
            # feed forward
            layer_output = softmax(np.dot(self.weights1, self.input))
            
            

            # back propagation


        
        

    def fit(self, epochs, learning_rate, validation_set=None):
        self.learning_rate = learning_rate
        history = dict(
            {
                "accuracy": [],
                "loss": [],
                "val_accuracy": [],
                "val_loss": []
            }
        )

        for i in range(epochs):
            self.feed_forward(self.input)

            _loss, _ = nll_loss(self.layer2, self.y)
            predicted = np.argmax(self.layer2, axis=1)
            _accuracy = accuracy_score(self.y, predicted)


            history["accuracy"].append(_accuracy)
            history["loss"].append(_loss)

            
            self.back_propagate()  

            #print(predicted)
            #print(self.y)

            validation_text = ""
            if validation_set:
                X_valid, y_valid = validation_set
                self.feed_forward(X_valid)
                loss, _ = nll_loss(self.layer2, y_valid)
                predictions = np.argmax(self.layer2, axis=1)
                accuracy = accuracy_score(y_valid, predictions)
                
                validation_text = f" -> Valid Acc: {round(accuracy, 5)} - Valid Loss: {round(loss, 5)}"

            sys.stdout.write(f"\rEpochs: {i+1} -> Accuracy: {round(_accuracy, 5)} - Loss: {round(_loss, 5)}{validation_text}")
            sys.stdout.flush()
            
        
        return history

    def predict(self, X):
        self.feed_forward(X)

        predictions = np.argmax(self.layer2, axis=1)

        return predictions

        

    def feed_forward(self, data):
        relu_vec = np.vectorize(relu)
        softmax_vec = np.vectorize(softmax)
        
        #print(data.shape)
        #print(self.weights1.shape)
        self.layer1 = relu_vec(np.dot(data, self.weights1))
        self.layer1 = self.layer1/np.max(self.layer1)

        self.layer2 = np.apply_along_axis(softmax, 1, np.dot(self.layer1, self.weights2))
        #print(self.layer2)

    def back_propagate(self):
        #print("back propagation")
        dscores = self.layer2
        #print(dscores, "\n**")
        dscores[range(len(self.y)), self.y] -= 1
        
        #print(dscores, "\n**")

        #dscores /= self.num_examples
        


        dw2 = np.dot(self.layer1.T, dscores)
        dhidden = np.dot(dscores, self.weights2.T)
        dhidden[self.layer1 <= 0] = 0
        dw = np.dot(self.input.T, dhidden)
        
        self.weights1 += -self.learning_rate*dw
        self.weights2 += -self.learning_rate*dw2


        #print("after")
        #print(self.weights2)

    def save_weights(self):
        root = "weights"
        if not os.path.exists(root):
            os.makedirs(root)
        

        np.save(os.path.join(root, "weights1.npy"), self.weights1)
        np.save(os.path.join(root, "weights2.npy"), self.weights2)
        

    def load_weights(self, filenames):
        # .npy file
        root = "weights"
        self.weights1 = np.load(os.path.join(root, filenames[0]))
        self.weights2 = np.load(os.path.join(root, filenames[1]))
            


    