import math
from collections import defaultdict, deque
class Val:
    def __init__(self, value, children=(), op="", label=""):
        self.value = value
        self.children = children
        self.op = op
        self.grad = 1
        self.label = label

        if self.op == "*":
            def _backward():
                self.children[0].grad += self.children[1].value * self.grad
                self.children[1].grad += self.children[0].value * self.grad

            self._backward = _backward

        elif self.op == "/":
            def _backward():
                self.children[0].grad += 1 / self.children[1].value * self.grad
                self.children[1].grad += (-self.children[0].value / (self.children[1].value * self.children[1].value)) * self.grad

            self._backward = _backward

        elif self.op == "tanh":
            def _backward():
                self.children[0].grad += (1 - (self.value * self.value)) * self.grad

            self._backward = _backward

    def __add__(self, other):
        return Val(self.value + other.value, (self,other), "+")

    def __mul__(self, other):
        return Val(self.value * other.value, (self,other), "*")

    def __sub__(self, other):
        return self + other.neg()

    def __truediv__(self, other):
        return Val(self.value / other.value, (self, other), "/")

    def neg(self):
        return self * Val(-1)

    def tanh(self):
        x = self.value
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        return Val(t, (self,), 'tanh')

    def _backward(self):
        return None
    def backward(self):
        self.grad = 1

        topo = self.topological_order()

        for node in topo:
            node._backward()

    def topological_order(self):
        in_edges = defaultdict(int)
        visited = set()
        def visit(node):
            for child in node.children:
                in_edges[child] += 1
                if child not in visited:
                    visited.add(child)
                    visit(child)

        queue = deque([])
        for node in in_edges:
            if in_edges[node] == 0:
                queue.append(node)

        result = []
        while len(queue):
            curr = queue.popleft()
            result.append(curr)

            for child in curr.children:
                in_edges[child] -= 1

                if in_edges[child] == 0:
                    queue.append(child)

        return result


    def __repr__(self):
        return f"Val({self.value})"



