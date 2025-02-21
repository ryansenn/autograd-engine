from graphviz_util import render
import math
class Val:
    def __init__(self, value, children=(), op=""):
        self.value = value
        self.children = children
        self.op = op
        self.grad = 0

    def __add__(self, other):
        return Val(self.value + other.value, (self,other), "+")

    def __mul__(self, other):
        return Val(self.value * other.value, (self,other), "*")

    def tanh(self):
        x = self.value
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Val(t, (self,), 'tanh')
        return out

    def backward(self):
        self.grad = 1
    def __repr__(self):
        return f"Val({self.value})"


# inputs x1,x2
x1 = Val(2.0)
x2 = Val(0.0)

# weights w1,w2
w1 = Val(-3.0)
w2 = Val(1.0)

# bias of the neuron
b = Val(6.8813735870195432)

# x1*w1 + x2*w2 + b
x1w1 = x1*w1
x2w2 = x2*w2

x1w1x2w2 = x1w1 + x2w2

n = x1w1x2w2 + b

o = n.tanh()

render(o)