from engine import Val
import random

class Neuron:
    def __init__(self, w_size):
        self.w = [Val(random.uniform(-w_size, w_size)) for i in range(w_size)]
        self.b = Val(random.uniform(-w_size, w_size))

    def __call__(self, x):
        result = self.b
        for i in range(len(x)):
            result += x[i] * self.w[i]

        return result


class Layer:
    def __init__(self, in_features, out_features):
        self.neurons = [Neuron(in_features) for i in range(out_features)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]


class Activation:
    def __call__(self, x):
        return [v.tanh() for v in x]

class Sequential:
    def __init__(self, *modules):
        self.modules = modules

    def __call__(self, x):
        for module in self.modules:
            x = module(x)

        return x
