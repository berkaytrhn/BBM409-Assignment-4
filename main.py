import numpy as np
import argparse

from sklearn.utils import validation
from neural_network import NeuralNetwork
from sklearn.metrics import accuracy_score
from dataloader import DataLoader
from utils import *
from itertools import product

def multi_experiment(batch_sizes, learning_rates, hidden_layers, data, dataloader):
    images, labels = data
    num_classes = len(np.unique(labels))
    prdct = list(product(batch_sizes, learning_rates, hidden_layers))
    epochs=25
    
    
    choosing_best = []

    for parameters in prdct:
        
        batch_size = parameters[0]
        lr = parameters[1]
        hidden_layer = parameters[2]
        print(batch_size, lr, hidden_layer)
        X_train, X_valid, X_test, y_train, y_valid, y_test = dataloader.batch_and_split_process(images, labels, batch_size, test_size=0.2, valid_size=0.2)

        nn = NeuralNetwork(X_train, y_train, hidden_layer_size=hidden_layer, output_size=num_classes)

        history = nn.fit(epochs, lr, validation_set=(X_valid, y_valid), save_every=10)
        accuracy = history["accuracy"]
        loss = history["loss"]

        save_graph("graphs", accuracy, "Accuracies", "Accuracy", f"acc_{lr}lr_{batch_size}bs_{hidden_layer}hl.png")
        save_graph("graphs", loss, "Losses", "Loss", f"loss_{lr}lr_{batch_size}bs_{hidden_layer}hl.png")

        pred = nn.predict(X_test)
        test_acc = round(accuracy_score(y_test, pred)*100, 5)
        choosing_best.append()
        print(f"\nTest accuracy: {test_acc}")

# mainde save plot dene sonra multi experiment çalıştır
def main(args):
    np.warnings.filterwarnings("ignore")

    
    BATCH_SIZE=32
    HEIGHT=32
    WIDTH=32
    HIDDEN_LAYERS=2
    EPOCHS = 50
    LEARNING_RATE = 0.01
    
    
    dataloader = DataLoader("./dataset", size=(HEIGHT, WIDTH))
    images, labels = dataloader.load_images()
    num_classes = len(np.unique(labels))
    X_train, X_valid, X_test, y_train, y_valid, y_test = dataloader.batch_and_split_process(images, labels, BATCH_SIZE, test_size=0.2, valid_size=0.2)

    nn = NeuralNetwork(X_train, y_train, hidden_layer_size=HIDDEN_LAYERS, output_size=num_classes)

    if args.load_weights:
        nn.load_weights()
    else:
        

        history = nn.fit(EPOCHS, LEARNING_RATE, validation_set=(X_valid, y_valid), save_every=10)
        #history = nn.fit(EPOCHS, LEARNING_RATE)
        accuracy = history["accuracy"]
        loss = history["loss"]
        plot_graph(accuracy, "Accuracies", "Accuracy")
        plot_graph(loss, "Losses", "Loss")
        save_graph("graphs", accuracy, "Accuracies", "Accuracy", f"acc_{LEARNING_RATE}lr_{BATCH_SIZE}bs_{HIDDEN_LAYERS}hl.png")
        nn.save_weights()

    pred = nn.predict(X_test)
    print(f"\nTest accuracy: {round(accuracy_score(y_test, pred)*100, 5)}")



    if args.experiment:
        batch_sizes = [16, 32, 64, 128]
        learning_rates = [0.005, 0.010, 0.015, 0.020]
        hidden_layers = [0, 1, 2]
        multi_experiment(batch_sizes, learning_rates, hidden_layers, [images, labels], dataloader)





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_weights", default=None, action="store_true")
    parser.add_argument("--experiment", default=None, action="store_true")


    args = parser.parse_args()


    main(args)

    



