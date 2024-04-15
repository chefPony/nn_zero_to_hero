import pytest
import math
from micrograd.micrograd import Value
from torch import Tensor


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
    b = a + 3
    c = a + b + b
    c.backward()
    amg, cmg = a, c

    a = Tensor([2.]).double()
    a.requires_grad = True
    b = a + 3
    c = a + b + b
    c.backward()
    apt, cpt = a, c

    assert cmg.data == cpt.item()
    assert amg.grad == apt.grad.item()


def test_backward_sub():
    a = Value(2)
    b = Value(-1)
    c = a - b
    c.backward()
    assert c.grad == 1
    assert a.grad == 1
    assert b.grad == -1


def test_backward_1():
    a = Value(2.)
    b = a - 3
    c = a * b + b * b
    d = c.leaky_relu(0.)
    d.backward()
    amg, dmg = a, d

    a = Tensor([2.]).double()
    a.requires_grad = True
    b = a - 3
    c = a * b + b * b
    d = c.relu()
    d.backward()
    apt, dpt = a, d

    assert dpt.item() == dmg.data
    assert amg.grad == apt.grad.item()


def test_backward_2():
    a = Value(2.)
    b = a - 5
    c = a * b + b * b
    z = a.exp()
    d = c.leaky_relu(0.) + z.tanh() + a/c
    d.backward()
    amg, dmg = a, d

    a = Tensor([2.]).double()
    a.requires_grad = True
    b = a - 5
    c = a * b + b * b
    z = a.exp()
    d = c.relu() + z.tanh() + a/c
    d.backward()
    apt, dpt = a, d

    assert dpt.item() == dmg.data
    assert amg.grad == apt.grad.item()


def test_backward_pow():
    a = Value(6)
    b = a ** 2.5
    b.backward()
    amg, bmg = a, b

    a = Tensor([6.])
    a.requires_grad = True
    b = a**2.5
    b.backward()
    apt, bpt = a, b
    assert abs(amg.grad - apt.grad.item()) < 1e-5
    assert abs(bmg.data - bpt.item()) < 1e-4
