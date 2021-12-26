import numpy as np

def nll_loss(probs, y):
    num_examples = len(y)
    #print(probs[range(num_examples), y])
    correct_log_probs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_log_probs)/num_examples
    return data_loss, correct_log_probs