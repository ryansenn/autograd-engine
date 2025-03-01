from graphviz_util import render
import math
class Val:
    def __init__(self, value, children=(), op="", label=""):
        self.value = value
        self.children = children
        self.op = op
        self.grad = 1
        self.label = label

    def __add__(self, other):
        return Val(self.value + other.value, (self,other), "+")

    def __mul__(self, other):
        return Val(self.value * other.value, (self,other), "*")

    def __sub__(self, other):
        return self + other.neg()

    def neg(self):
        return self * Val(-1)
    def tanh(self):
        x = self.value
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        return Val(t, (self,), 'tanh')

    def backward(self):
        if self.op == "":
            return None

        if self.op == "*":
            self.children[0].grad = self.children[1].value
            self.children[1].grad = self.children[0].value

        if self.op == "tanh":
            self.children[0].grad = 1 - (self.value * self.value)

        for child in self.children:
            child.grad *= self.grad
            child.backward()



    def __repr__(self):
        return f"Val({self.value})"


x = Val(2.0, label="x")
w = Val(3.0, label="w")
b = Val(-3.0, label="b")

y = (w*x + b).tanh()
y.label="y"

l = Val(40) - y
l.label="l"

l.backward()

render(l)

