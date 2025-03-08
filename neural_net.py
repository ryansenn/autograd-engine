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

    def parameters(self):
        for weight in self.w:
            yield weight

        yield self.b

    def __repr__(self):
        return "w: " + str(self.w) + " b: " + str(self.b)


class Layer:
    def __init__(self, in_features, out_features):
        self.neurons = [Neuron(in_features) for i in range(out_features)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        for neuron in self.neurons:
            yield from neuron.parameters()


class Activation:
    def __call__(self, x):
        return [v.tanh() for v in x]


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        for layer in self.layers:
            yield from layer.parameters()



class Loss:
    def __call__(self, model, y):
        loss = (y[0] - model[0]) * (y[0] - model[0])
        for i in range(1,len(model)):
            loss += (y[i] - model[i]) * (y[i] - model[i])

        return loss / Val(len(model))


class SGD:
    def __init__(self, params, lr=0.1):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.value = param.value - (self.lr*param.grad)

    def zero_grad(self):
        for param in self.params:
            param.grad = 0
