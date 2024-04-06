class Value:

    def __init__(self, data, _children=(), _op=None):

        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, _children=(self, other), _op="ADD")
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, _children=(self, other), _op="MUL")
        return out

    def _backward(self, x):
        if not self._prev or (self == x):
            return float(self == x)
        if self._op == "ADD":
            return sum([c._backward(x) for c in self._prev])
        elif self._op == "MUL":
            return self._prev[0]._backward(x) * self._prev[1].data + \
                self._prev[0].data * self._prev[1]._backward(x)

    def backward(self, _level=0):
        self.grad = 1.0 if _level == 0 else self.grad
        s = (3 - len(self._prev)) * sum([x.data for x in self._prev])
        if self._op == "ADD":
            for x in self._prev:
                x.grad += self.grad * (3 - len(self._prev))
                x.backward(_level+1)
        elif self._op == "MUL":
            for x in self._prev:
                x.grad += self.grad * (3 - len(self._prev)) * (s - x.data)
                x.backward(_level+1)


    #@property
    #def grad(self):
    #    return self.next.backward(self)
