from neural_net import *
from graphviz_util import render


layer = Layer(2,1)
x = [Val(2), Val(2)]
print(layer(x))
render(layer(x)[0])

