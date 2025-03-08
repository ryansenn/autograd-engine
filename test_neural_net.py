from neural_net import *
from visualizer.graphviz_util import render
import random
import numpy as np

# Generate Data
data_size = 1000
x_raw = np.array([random.uniform(-100, 100) for _ in range(data_size)])
y_raw = np.array([(2 * (val ** 2)) + (7 * val) + 12 for val in x_raw])

# Normalize Input and Output
x_mean, x_std = np.mean(x_raw), np.std(x_raw)
y_mean, y_std = np.mean(y_raw), np.std(y_raw)

x_norm = (x_raw - x_mean) / x_std
y_norm = (y_raw - y_mean) / y_std

x = [[Val(x_norm[i])] for i in range(data_size)]
y = [[Val(y_norm[i])] for i in range(data_size)]

# Test Predictions (Denormalizing Outputs)
def predict(model_, val):
    val_norm = (val - x_mean) / x_std
    prediction_norm = model_([Val(val_norm)])[0].value
    return (prediction_norm * y_std) + y_mean  # Denormalized output

# Model
model = Sequential(
    Layer(1, 8),
    Activation(),
    Layer(8, 1)
)

criterion = Loss()
optimizer = SGD(model.parameters(), lr=0.01)

# Create separate test data
test_size = 200
x_test_raw = np.array([random.uniform(-100, 100) for _ in range(test_size)])
y_test_raw = np.array([2 * (val ** 2) + 7 * val + 12 for val in x_test_raw])

# Function to calculate MSE
def mean_squared_error(model_, x_data, y_data):
    preds = np.array([predict(model_, val) for val in x_data])
    return np.mean((preds - y_data) ** 2)

# Training Loop with Convergence Check
prev_loss = float('inf')
threshold = 1e-5  # Minimum loss change threshold

for epoch in range(500):
    total_loss = 0
    for i in range(len(x)):
        optimizer.zero_grad()
        output = model(x[i])
        loss = criterion(output, y[i])
        loss.backward()
        optimizer.step()
        total_loss += loss.value  # Accumulate loss

    avg_loss = total_loss / data_size
    test_mse = mean_squared_error(model, x_test_raw, y_test_raw)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss}, Test MSE: {test_mse}")

    prev_loss = avg_loss

print(predict(model, 1))
print(predict(model, 2))
print(predict(model, 10))
print(predict(model, -5))

# Compute final MSE on test data
mse = mean_squared_error(model, x_test_raw, y_test_raw)
rmse = np.sqrt(mse)

print(f"Final Test MSE: {mse}")
print(f"Final Test RMSE: {rmse}")