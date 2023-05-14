import numpy as np


class Layer:

    def __init__(self, m, n=None):
        """ m: neurons of the current layer. n: neurons of the previous layer
        """
        self.m, self.n = m, n
        self.w = np.random.randn(m, n)
        self.b = np.random.randn(m, 1)

    def flush_gradients(self):
        self.dw = np.zeros((self.m, self.n))
        self.db = np.zeros((self.m, 1))


class NeuralNet:

    def __init__(self, layers):
        """ layers can be of the form [x, y, z], meaning first layer has x neurons
        second has y, and so on. First layer is the input layer
        """
        self.n_layers = len(layers)  # first layer in input
        self.layers = [Layer(layers[i], layers[i-1] if i > 0 else 1)
                       for i in range(self.n_layers)]

    def feedforward(self, a):
        for layer in self.layers[1:]:
            a = sigmoid(np.dot(layer.w, a) + layer.b)
        return a

    def train(self, train_data, val_data, epochs, batchsize, eta):

        for i in range(epochs):
            np.random.shuffle(train_data)
            for j in range(0, len(train_data), batchsize):
                batch = train_data[j: j+batchsize]
                self.update_wb(batch, eta)

            accuracy = np.mean([np.argmax(self.feedforward(x)) == y for x, y in val_data])
            print(f"Epoch {i}: Accuracy: {accuracy*100: .2f} %")

    def update_wb(self, batch, eta):
        for l in self.layers:
            l.flush_gradients()  # flushed, then later updated in backprop

        # Accumulate gradients using back propagation
        for x, y in batch:
            self.back_propagation(x, y)  # updates layer graditents with dw, db

        for l in self.layers[1:]:  # update weights
            l.w = l.w - l.dw / len(batch) * eta
            l.b = l.b - l.db / len(batch) * eta

    def back_propagation(self, x, y):

        # forward pass, first layer is input
        self.layers[0].a = a = x
        for l in self.layers[1: ]:
            z = np.dot(l.w, a) + l.b
            a = sigmoid(z)
            l.z, l.a = z, a

        # backward pass. Use Quadratic loss function
        for i in range(self.n_layers-1, 0, -1): # go from last layer till 1
            if i == self.n_layers-1:  # last layer
                self.layers[i].delta = (self.layers[i].a - y) * sigmoid_gradient(self.layers[i].z)
            else:
                self.layers[i].delta = np.dot(self.layers[i+1].w.T, self.layers[i+1].delta) * sigmoid_gradient(self.layers[i].z)
            self.layers[i].db += self.layers[i].delta
            self.layers[i].dw += np.dot(self.layers[i].delta, self.layers[i-1].a.transpose())


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))
