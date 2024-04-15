from typing import Tuple
import numpy as np
from micrograd.micrograd import Value


class Neuron:

    def __init__(self, dim):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(dim)]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, X):
        out = sum([x * self.w[i] for i, x in enumerate(X)]) + self.b
        return out

    def __repr__(self):
        return f"Neuron(w={self.w}, b={self.b})"

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for w in self.w:
            w.grad = 0
        self.b.grad = 0

    def update(self, lr: float):
        self.w = [Value(data=w.data - lr * w.grad) for w in self.w]
        self.b = Value(self.b.data - lr * self.b.grad)


class Layer:

    def __init__(self, nin, nout, activation):
        self._activations = {"tanh": Value.tanh, "relu": Value.leaky_relu, "sigmoid": Value.sigmoid}
        self.neurons = [Neuron(nin) for i in range(nout)]
        self.activation = self._activations[activation]

    def __call__(self, X):
        out = [self.activation(n(X)) for n in self.neurons]
        out = out if len(out) > 1 else out[0]
        return out

    def parameters(self):
        parameters = list()
        for n in self.neurons:
            parameters.extend(n.parameters())
        return parameters

    def zero_grad(self):
        for n in self.neurons:
            n.zero_grad()

    def update(self, lr):
        for n in self.neurons:
            n.update(lr)


class MicroNetwork:

    def __init__(self, nin: int, nouts: Tuple[int, str]):
        input_dim = nin

        self.layers = []
        for nou, activation in nouts:
            self.layers.append(Layer(input_dim, nou, activation))
            input_dim = nou

    def __call__(self, X):
        out = X
        for l in self.layers:
            out = l(out)
        return out

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.

    def update(self, lr):
        for l in self.layers:
            l.update(lr)
