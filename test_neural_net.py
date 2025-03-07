from neural_net import *
from graphviz_util import render

model = Sequential(Layer(1,1))

x = [[Val(i)] for i in range(10)]
y = [[Val(i*2+5)] for i in range(10)]

criterion = Loss()
optimizer = SGD(model.parameters(), lr=0.1)

print(model.layers[0].neurons[0])

for epoch in range(10):
    for i in range(len(x)):
        optimizer.zero_grad()
        output = model(x[i])
        loss = criterion(output, y[i])
        loss.backward()
        optimizer.step()
        print(model.layers[0].neurons[0])


