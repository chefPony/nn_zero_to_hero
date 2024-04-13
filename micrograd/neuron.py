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

    def reset_grad(self):
        for w in self.w:
            w.grad = 0
        self.b.grad = 0

    def update(self, lr: float):
        self.w = [Value(data=w.data - lr * w.grad) for w in self.w]
        self.b = Value(self.b.data - lr * self.b.grad)