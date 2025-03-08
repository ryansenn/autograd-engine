from neural_net import *
from visualizer.graphviz_util import render

model = Sequential(Layer(1,1))
data_size = 100
x = [[Val(i)] for i in range(data_size)]
y = [[Val(2*i+5)] for i in range(data_size)]

criterion = Loss()
optimizer = SGD(model.parameters(), lr=0.1)

for epoch in range(10):
    for i in range(len(x)):
        optimizer.zero_grad()
        output = model(x[i])
        loss = criterion(output, y[i])
        loss.backward()
        optimizer.step()

    print(str(model.layers[0].neurons[0]) + " r: " + str(output[0]) + " l:" + str(loss))








