from graphviz_util import render
import math
class Val:
    def __init__(self, value, children=(), op=""):
        self.value = value
        self.children = children
        self.op = op
        self.grad = 1

    def __add__(self, other):
        return Val(self.value + other.value, (self,other), "+")

    def __mul__(self, other):
        return Val(self.value * other.value, (self,other), "*")

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

        for child in self.children:
            child.grad *= self.grad
            child.backward()



    def __repr__(self):
        return f"Val({self.value})"


x = Val(2.0)
w = Val(3.0)
b = Val(-3.0)

y = w*x + b

y.backward()

render(y)

