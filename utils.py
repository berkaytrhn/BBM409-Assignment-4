import numpy as np
import matplotlib.pyplot as plt
import os


def plot_graph(data, title, value):
    _plot = plt.plot(range(len(data)), data)
    plt.title(title)
    plt.legend([_plot], [value])
    plt.xlabel("Epoch")
    plt.ylabel(value)
    plt.show()

def save_graph(root_dir, data, title, value, file_name):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    _plot = plt.plot(range(len(data)), data)
    plt.title(title)
    plt.legend([_plot], [value])
    plt.xlabel("Epoch")
    plt.ylabel(value)
    plt.savefig(os.path.join(root_dir, file_name))
    plt.clf()
    plt.cla()