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
    np.warnings.filterwarnings("ignore")

    data = sns.load_dataset("iris")
    
    data["species"] = LabelEncoder().fit_transform(data["species"])
    BATCH_SIZE=32
    dataloader = DataLoader("./dataset", BATCH_SIZE)
    #dataloader.preprocess()
    image_X, image_y = dataloader.load_images()
    
    """
    data = np.array(data)
    np.random.shuffle(data)
    X = data[:,:-1]
    y = data[:,-1].astype(int)
    """


    # test set
    X_train, X_test, y_train, y_test = train_test_split(image_X, image_y, shuffle=True, test_size=0.2)

    # validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, shuffle=True, test_size=0.2)




    nn = NeuralNetwork(X_train, y_train)
    #nn = NeuralNetwork(image_X, image_y)


    if args.load_weights:
        nn.load_weights(["weights1.npy", "weights2.npy"])



    else:
        EPOCHS = 10000
        LEARNING_RATE = 0.0004

        history = nn.fit(EPOCHS, LEARNING_RATE, validation_set=(X_valid, y_valid))
        #history = nn.fit(EPOCHS, LEARNING_RATE)
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

    



