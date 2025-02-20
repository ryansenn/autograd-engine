class Val:
    def __init__(self, value, children=(), op=""):
        self.value = value
        self.children = children
        self.op = op

    def __add__(self, other):
        return Val(self.value + other.value, (self,other), "+")

    def __mul__(self, other):
        return Val(self.value * other.value, (self,other), "*")

    def __repr__(self):
        return f"Val({self.value})"


a = Val(2)
b = Val(5)
c = Val(12)

print(a * b + c)