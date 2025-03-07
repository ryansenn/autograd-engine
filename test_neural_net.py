from neural_net import *
from graphviz_util import render

model = Sequential(Layer(1,5),
                   Activation(),
                   Layer(5,1))

x = [Val(2)]

criterion = Loss()

print(model(x))
#render(model(x)[0])

