import pytest
from micrograd.micrograd import Value


def test_add():
    a = Value(2)
    b = Value(3)
    c = a + b
    assert c.data == 5
    assert c._prev == {a, b}
    assert c._op == "ADD"


def test_mul():
    a = Value(2)
    b = Value(3)
    c = a * b
    assert c.data == 6
    assert c._prev == {a, b}
    assert c._op == "MUL"


def test_backward_sum():
    a = Value(2)
    b = Value(-1)
    c = a + b + b
    c.backward()
    assert c.grad == 1
    assert a.grad == 1
    assert b.grad == 2


def test_backward_mul():
    a = Value(-1)
    b = Value(2)
    c = a * b + b * b
    c.backward()
    assert c.grad == 1
    assert a.grad == 2
    assert b.grad == 3


def test_backward():
    # g = e * d = (d + a) * (c + b * b) = (c + b^2 + a) * (c + b^2) = c^2 + 2cb^2 + b^4 +ac + ab^2
    # dg/db = 4cb + 4b^3 + 2ab = 8 + 4 + 4 = 16
    # dg/dd = (d + a) * d = d^2 + ad = 2d + a = 6 + 2 = 8
    # dg/dc = (d + a) * d = (c + b^2 + a) * (c + b^2) = c^2 + 2cb^2 + b^4 + ac +ab^2 = 2c + 2b^2 + a= 4 + 2 + 2
    a, b = Value(2, label="a"), Value(1, label="b")
    c = a + b      # 3
    c.label = "c"
    d = a * b      # 2
    d.label = "d"
    e = c + d      # 5
    e.label = "e"
    g = e * d      # 10
    g.label = "g"
    g.backward()
    assert g.grad == 1
    assert e.grad == 2
    assert d.grad == 7
    assert c.grad == 2
    assert b.grad == 16
    assert a.grad == 9

