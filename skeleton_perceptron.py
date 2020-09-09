#################################
# Your name: shay fux
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""


def helper():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def perceptron(data, labels):
    """
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    # TODO: Implement me
    d = len(data[0])
    w = np.array([0 for i in range(d)])
    for t in range(len(data)):
        xt = normilize(data[t])  # normilize the t smaple in each iteration
        predictor = sign(w, xt)
        if predictor != labels[t]:
            w = w + labels[t] * xt
    return w

    #################################

    # Place for additional code
def sign(w, x):  # w and x are nd.arrays
    dot_product = np.dot(w, x)
    if dot_product >= 0:
        return 1
    else:
        return -1


def normilize(x):  # x is nd.array
    if zero_vec_check(x) == True:  # if x is the zero vec return him
        return x
    norma = (np.dot(x, x)) ** 0.5
    x = (1 / norma) * x
    return x


def zero_vec_check(x):
    for p in x:
        if p != 0:
            return False
    return True

def shuffle_sample(data, labels, n_size):
    lst = [[data[i], labels[i]] for i in range(n_size)]
    vec = np.array(lst)
    np.random.shuffle(vec)
    new_data = [vec[i][0] for i in range(n_size)]
    new_lables = [vec[i][1] for i in range(n_size)]
    return new_data, new_lables


def accuracy_precentage(w, data, labels):
    counter = 0
    n = len(data)
    for i in range(n):
        x = np.array(data[i])
        predictor = sign(w, x)
        if predictor == labels[i]:
            counter = counter + 1
    return (counter / n)

def miss_classfied(w, data, labels):
    n = len(data)
    for i in range(n):
        x = np.array(data[i])
        predictor = sign(w, x)
        if predictor != labels[i]:
            return x

# (a)
def sec_a():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    nlst = [5, 10, 50, 100, 500, 1000, 5000]
    accuracy_lst = []
    for n in nlst:
        tmp_accuracy_lst = []
        for m in range(100):
            n_train_data, n_train_labels = shuffle_sample(train_data, train_labels, n)
            w = perceptron(n_train_data, n_train_labels)
            tmp_accuracy_lst.append(accuracy_precentage(w, test_data, test_labels))
        accuracy_lst.append(np.array(tmp_accuracy_lst))
        print(w.shape)
    mean_lst = [numpy.mean(accuracy_lst[i]) for i in range(len(accuracy_lst))]
    percentile_5_lst = [numpy.percentile(accuracy_lst[i], 5) for i in range(len(accuracy_lst))]
    percentile_95_lst = [numpy.percentile(accuracy_lst[i], 95) for i in range(len(accuracy_lst))]
sec_a()

# (b)
def sec_b():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    w = perceptron(train_data,train_labels)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()

#sec_b()


def sec_c():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    w = perceptron(train_data ,train_labels)
    print(accuracy_precentage(w, test_data, test_labels))

#sec_c()

def sec_d():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    w = perceptron(train_data, train_labels)
    image = miss_classfied(w, test_data, test_labels)
    plt.imshow(np.reshape(image, (28, 28)), interpolation='nearest')
    plt.show()

#sec_d()


#################################
