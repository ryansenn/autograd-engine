## Autograd

My implementation of a minimal autograd and neural net library, inspired by karpathy’s micrograd. 

### Example

```python
from neural_net import *

model = Sequential(Layer(1, 8),
                   Activation(),
                   Layer(8, 1))

optimizer = SGD(model.parameters(), lr=0.01)

x, y = [Val(0.5)], [Val(0.8)]

for epoch in range(100):
    optimizer.zero_grad()
    loss = Loss()(model(x), y)
    loss.backward()
    optimizer.step()

print("Prediction:", model(x)[0].value)
