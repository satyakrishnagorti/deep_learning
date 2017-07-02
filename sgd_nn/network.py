# Practice from Michael Nielson's chaprter 1
import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_delta(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Network:

    def __init__(self, sizes):

        self.num_of_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    

    def feedforward(self, a):

        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

        if test_data: n_test = len(test_data)
        n = len(training_data)
        random.shuffle(training_data)

        for e in range(epochs):
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.train_mini_batch(mini_batch, eta)
                
            if test_data:
                print("Epoch {0}: {1} / {2}".format(e, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(e))
    
    def train_mini_batch(self, mini_batch, eta):

        biases_total_delta = [np.zeros(b.shape) for b in self.biases]
        weights_total_delta = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            b_mini_delta, w_mini_delta = self.backprop(x, y)

            biases_total_delta = [btd + bmd for btd, bmd in zip(biases_total_delta, b_mini_delta)]
            weights_total_delta = [wtd + wmd for wtd, wmd in zip(weights_total_delta, w_mini_delta)]
            self.weights = [w - (eta / len(mini_batch)) * wtd for w, wtd in zip(self.weights, weights_total_delta)]
            self.biases = [b - (eta / len(mini_batch)) * btd for b, btd in zip(self.biases, biases_total_delta)]
        
    def backprop(self, x, y):
        b_mini_delta  = [np.zeros(b.shape) for b in self.biases]
        w_mini_delta  = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        # forward propagation
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # backward propagation
        delta = (activations[-1] - y) * sigmoid_delta(zs[-1])

        b_mini_delta[-1] = delta
        w_mini_delta[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_of_layers):
            z = zs[-l]
            sd = sigmoid_delta(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sd
            b_mini_delta[-l] = delta
            w_mini_delta[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (b_mini_delta, w_mini_delta)

    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)