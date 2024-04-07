import math


class Value:

    def __init__(self, data, label='', _children=(), _op=None):

        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda : None
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label}])"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op="ADD")

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="MUL")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, _children=(self, other), _op="SUB")

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * (-1.0)
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), _children=(self,), _op="EXP")

        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Power operation compatible only with integers and float")
        out = Value(self.data ** other, _children=(self, ))

        def _backward():
            self.grad = out.grad * other * self.data ** (other - 1)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = self * (other ** -1)
        return out

    def tanh(self):
        t = (math.exp(2 * self.data) - 1)/(math.exp(2 * self.data) + 1)
        out = Value(t, _children=(self, ), _op="TANH")

        def _backward():
            self.grad += out.grad * (1 - out.data ** 2)

        out._backward = _backward
        return out

    def backward(self):
        # Topological sort
        height = 0
        queue = [(x, height + 1) for x in self._prev]
        order = {self: 0}
        while queue:
            target, height = queue.pop()
            current_height = max(height, order.get(target, 0))
            order[target] = current_height
            queue += [(child, current_height + 1) for child in target._prev]
        #backward propagation
        self.grad = 1
        order = sorted(order, key=lambda x: order.get(x))
        for x in order:
            x._backward()