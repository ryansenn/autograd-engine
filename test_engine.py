import torch
from engine import Val
from graphviz_util import render

# PyTorch implementation
x_torch = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
w_torch = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
b_torch = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
c_torch = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)

y_torch = (w_torch * x_torch + b_torch) / c_torch
o_torch = torch.tanh(y_torch)

o_torch.backward()

# Custom implementation
x_val = Val(2.0, label="x")
w_val = Val(3.0, label="w")
b_val = Val(-3.0, label="b")
c_val = Val(4.0, label="c")

y_val = ((w_val * x_val + b_val) / c_val)
y_val.label = "y"

y_val.backward()

tol = 1e-6

def check_close(name, torch_val, custom_val):
    match = abs(torch_val - custom_val) < tol
    status = "Match" if match else "Mismatch"
    print(f"{name}: PyTorch={torch_val}, Custom={custom_val} -> {status}")

# Compare output values and gradients
check_close("Output", o_torch.item(), y_val.value)
check_close("Gradient x", x_torch.grad.item(), x_val.grad)
check_close("Gradient w", w_torch.grad.item(), w_val.grad)
check_close("Gradient b", b_torch.grad.item(), b_val.grad)
check_close("Gradient c", c_torch.grad.item(), c_val.grad)  # Added check for c

render(y_val)