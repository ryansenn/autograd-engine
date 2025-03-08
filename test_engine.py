import pytest
import torch
from engine import Val
from graphviz_util import render

# Tolerance for numerical comparison
TOL = 1e-6

@pytest.fixture
def torch_computation():
    x_torch = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    w_torch = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
    c_torch = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)

    y_torch = (w_torch * x_torch + b_torch) / c_torch
    o_torch = torch.tanh(y_torch)

    o_torch.backward()

    return x_torch, w_torch, b_torch, c_torch, o_torch

@pytest.fixture
def custom_computation():
    x_val = Val(2.0, label="x")
    w_val = Val(3.0, label="w")
    b_val = Val(-3.0, label="b")
    c_val = Val(4.0, label="c")

    y_val = ((w_val * x_val + b_val) / c_val).tanh()
    y_val.label = "y"

    y_val.backward()

    return x_val, w_val, b_val, c_val, y_val

def test_output(torch_computation, custom_computation):
    _, _, _, _, o_torch = torch_computation
    _, _, _, _, y_val = custom_computation

    assert abs(o_torch.item() - y_val.value) < TOL, f"Output mismatch: {o_torch.item()} vs {y_val.value}"

def test_gradients(torch_computation, custom_computation):
    x_torch, w_torch, b_torch, c_torch, _ = torch_computation
    x_val, w_val, b_val, c_val, _ = custom_computation

    assert abs(x_torch.grad.item() - x_val.grad) < TOL, f"Gradient x mismatch: {x_torch.grad.item()} vs {x_val.grad}"
    assert abs(w_torch.grad.item() - w_val.grad) < TOL, f"Gradient w mismatch: {w_torch.grad.item()} vs {w_val.grad}"
    assert abs(b_torch.grad.item() - b_val.grad) < TOL, f"Gradient b mismatch: {b_torch.grad.item()} vs {b_val.grad}"
    assert abs(c_torch.grad.item() - c_val.grad) < TOL, f"Gradient c mismatch: {c_torch.grad.item()} vs {c_val.grad}"

import pytest
import torch
from engine import Val

TOL = 1e-6

def test_complex_equation():
    # PyTorch setup
    x_torch = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    w_torch = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)
    c_torch = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)

    # More complex expression reusing variables, no exponent
    y_torch = w_torch * x_torch + b_torch
    z_torch = (x_torch + b_torch) / c_torch
    o_torch = w_torch * z_torch + c_torch * x_torch
    final_torch = y_torch + z_torch + torch.tanh(o_torch)
    final_torch.backward()

    # Custom engine setup
    x_val = Val(2.0, label="x")
    w_val = Val(3.0, label="w")
    b_val = Val(-1.0, label="b")
    c_val = Val(4.0, label="c")

    # Same expression in custom autograd
    y_val = w_val * x_val + b_val
    z_val = (x_val + b_val) / c_val
    o_val = w_val * z_val + c_val * x_val
    final_val = y_val + z_val + (o_val).tanh()
    final_val.backward()

    render(final_val)

    # Compare outputs
    assert abs(final_torch.item() - final_val.value) < TOL, \
        f"Output mismatch: {final_torch.item()} vs {final_val.value}"

    # Compare gradients
    assert abs(x_torch.grad.item() - x_val.grad) < TOL, \
        f"x grad mismatch: {x_torch.grad.item()} vs {x_val.grad}"
    assert abs(w_torch.grad.item() - w_val.grad) < TOL, \
        f"w grad mismatch: {w_torch.grad.item()} vs {w_val.grad}"
    assert abs(b_torch.grad.item() - b_val.grad) < TOL, \
        f"b grad mismatch: {b_torch.grad.item()} vs {b_val.grad}"
    assert abs(c_torch.grad.item() - c_val.grad) < TOL, \
        f"c grad mismatch: {c_torch.grad.item()} vs {c_val.grad}"