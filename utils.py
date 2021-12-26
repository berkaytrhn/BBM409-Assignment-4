import numpy as np
import matplotlib.pyplot as plt


def plot_graph(data, title, value):
    _plot = plt.plot(range(len(data)), data)
    plt.title(title)
    plt.legend([_plot], [value])
    plt.xlabel("Epoch")
    plt.ylabel(value)
    plt.show()