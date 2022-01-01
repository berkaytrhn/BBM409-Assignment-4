import numpy as np
from seaborn.rcmod import reset_defaults
from sklearn.utils import validation
from activations import *
from losses import *
from utils import *
import sys
from sklearn.metrics import accuracy_score
import os

class NeuralNetwork:

    def __init__(self, X, y, hidden_layer_size, output_size):
        self.input = X
        self.y = y
        self.output_size = output_size   
        self.hidden_layer_size=hidden_layer_size
        self.num_examples = len(y)

        sizes = []
        size = 2**((self.output_size//2)+2) # 128

        for i in range(1, self.hidden_layer_size+1):
            sizes.append(size*i)
        print(f"sizes: {sizes}")
        


        output_sizes = [
            self.output_size,
            *sizes
        ]

        input_sizes = [
            self.input[0].shape[1],
            *sizes
        ]
        output_sizes = sorted(output_sizes, reverse=True)
        input_sizes = sorted(input_sizes, reverse=True)

        self.weights = []   
        for i in range(len(output_sizes)):
            self.weights.append(np.random.rand(input_sizes[i], output_sizes[i]))

        self.biases = []
        for i in range(len(output_sizes)):
            self.biases.append(np.random.rand())

        self.layers = [
            self.input,
            0,
            0,
            0
        ]

        
        

    def fit(self, epochs, learning_rate, validation_set=None, save_every=10):
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
            if (epochs+1)%save_every==0:
                self.save_weights()
            
            self.batch_accuracies = []
            self.batch_losses = []
            self.batch_weight_derivatives = []
            self.batch_bias_derivatives = []

            for index in range(len(self.input)):# batch processing
                #num_samples = data.shape[0]
                _X = self.input[index]
                _y = self.y[index]
                self.feed_forward(_X)

                _loss, _ = nll_loss(self.layers[self.hidden_layer_size+1], _y)
                predicted = np.argmax(self.layers[self.hidden_layer_size+1], axis=1)
                _accuracy = accuracy_score(_y, predicted)

                self.batch_losses.append(_loss)
                self.batch_accuracies.append(_accuracy)

                self.back_propagate(_y)  

            weight_derivatives = np.sum(self.batch_weight_derivatives, axis=0)/len(self.input)
            bias_derivatives = np.sum(self.batch_bias_derivatives, axis=0)/len(self.input)
            _accuracy = np.mean(self.batch_accuracies)
            _loss = np.mean(self.batch_losses)
            self.update_weights_and_biases(weight_derivatives, bias_derivatives)


            history["accuracy"].append(_accuracy)
            history["loss"].append(_loss)
            #print(predicted)
            #print(self.y)

            validation_text = ""
            if validation_set:
                X_valid, y_valid = validation_set
                val_probs = self.validation_pass(X_valid)
                loss, _ = nll_loss(val_probs, y_valid)
                predictions = np.argmax(val_probs, axis=1)
                #print(y_valid.shape)
                #print(predictions.shape)
                accuracy = accuracy_score(y_valid, predictions)
                
                validation_text = f" -> Valid Acc: {round(accuracy, 5)} - Valid Loss: {round(loss, 5)}"
                
                history["val_accuracy"].append(accuracy)
                history["val_loss"].append(loss)


            sys.stdout.write(f"\rEpochs: {i+1} -> Accuracy: {round(_accuracy, 5)} - Loss: {round(_loss, 5)}{validation_text}")
            sys.stdout.flush()
            
        
        return history


    def validation_pass(self, X_valid):
        relu_vec = np.vectorize(relu)

        result = X_valid


        for i in range(1, self.hidden_layer_size+2):
            if i != self.hidden_layer_size+1:
                result = relu_vec(np.dot(result, self.weights[i-1]))
                result = result/np.max(result)
            else:
                result = np.apply_along_axis(softmax, 1, np.dot(result, self.weights[i-1])) # same layer weight, previous layer activations

        return result


    def predict(self, X):
        result = self.validation_pass(X)

        predictions = np.argmax(result, axis=1)

        return predictions

        

    def feed_forward(self, data):
        relu_vec = np.vectorize(relu)
        
        #print(data.shape)
        #print(self.weights1.shape)
        self.layers[0] = data # for batch processing

        for i in range(1, self.hidden_layer_size+2):
            if i != self.hidden_layer_size+1:
                self.layers[i] = relu_vec(np.dot(self.layers[i-1], self.weights[i-1])  + self.biases[i-1])
                self.layers[i] = self.layers[i]/np.max(self.layers[i])
            else:
                self.layers[i] = np.apply_along_axis(softmax, 1, np.dot(self.layers[i-1], self.weights[i-1]) + self.biases[i-1]) # same layer weight and bias, previous layer activations


    def back_propagate(self, _y):
        #print("back propagation")
        dscores = self.layers[self.hidden_layer_size+1]
        #print(dscores, "\n**")
        dscores[range(len(_y)), _y] -= 1
        
        #print(dscores, "\n**")
        dscores /= self.num_examples
        
        weight_derivatives = []

        bias_derivatives = []


        """
            2
            i = 0, 1, 2
            k= 2, 1, 0
            1
            i = 0, 1
            k= 1, 2
        """
        dcumulative = dscores
        
        for i in range(self.hidden_layer_size+1):
            dw_temp = np.dot(self.layers[self.hidden_layer_size-i].T, dcumulative)
            weight_derivatives.append(dw_temp)
            #print(sum(dcumulative))
            bias_derivatives.append(sum(dcumulative))
            dcumulative = np.dot(dcumulative, self.weights[self.hidden_layer_size-i].T)
            dcumulative[self.layers[self.hidden_layer_size-i] <= 0] = 0
 

        self.batch_weight_derivatives.append(np.array(weight_derivatives))
        self.batch_bias_derivatives.append(np.array(bias_derivatives))
        
        #reverse weight_derivatives array
        
        #print("after")
        #print(self.weights2)

    def update_weights_and_biases(self, weight_derivatives, bias_derivatives):
        
        weight_derivatives = weight_derivatives[::-1]      
        for i in range(self.hidden_layer_size+1):
            self.weights[i] += -self.learning_rate*weight_derivatives[i]

        
        # reverse bias_derivatives array
        bias_derivatives = bias_derivatives[::-1]
        for i in range(self.hidden_layer_size+1):
            
            self.biases[i] += -self.learning_rate*bias_derivatives[i]
        

    def save_weights(self):
        print("\nSaving Weights...")
        root = "weights"
        if not os.path.exists(root):
            os.makedirs(root)
        

        for i in range(self.hidden_layer_size+1):
            np.save(os.path.join(root,f"weight_{i}.npy"),self.weights[i])
        

    def load_weights(self):
        # .npy file
        root = "weights"
        
        for i in range(self.hidden_layer_size+1):
            self.weights[i] = np.load(os.path.join(root, f"weight_{i}.npy"))
            


    