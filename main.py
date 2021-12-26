import numpy as np
import argparse
from neural_network import NeuralNetwork
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dataloader import DataLoader
from utils import *

def main(args):

    data = sns.load_dataset("iris")
    
    data["species"] = LabelEncoder().fit_transform(data["species"])
    
    dataloader = DataLoader("./dataset")
    #dataloader.preprocess()
    #dataloader.load_images()

    data = np.array(data)
    np.random.shuffle(data)
    X = data[:,:-1]
    y = data[:,-1].astype(int)
    # test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)




    nn = NeuralNetwork(X_train, y_train)

    if args.load_weights:
        nn.load_weights("weights.npy")



    else:
        EPOCHS = 20000
        LEARNING_RATE = 0.0004

        history = nn.fit(EPOCHS, LEARNING_RATE)
        accuracy = history["accuracy"]
        loss = history["loss"]
        plot_graph(accuracy, "Accuracies", "Accuracy")
        plot_graph(loss, "Losses", "Loss")

        nn.save_weights()

    pred = nn.predict(X_test)
    print(f"\nTest accuracy: {round(accuracy_score(y_test, pred)*100, 5)}")






if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_weights", default=None)


    args = parser.parse_args()


    main(args)

    



