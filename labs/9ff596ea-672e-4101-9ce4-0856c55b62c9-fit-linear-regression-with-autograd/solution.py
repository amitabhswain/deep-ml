import torch
import numpy as np


def fit_linear_regression(X, y, lr=0.1, steps=500):
    """Fit y ~= X @ w + b with full-batch GD using only autograd."""
    
    # Ensure y is shape (N,), not (N,1)
    y = y.reshape(-1)
    
    N, D = X.shape
    
    # Initialize parameters with requires_grad=True
    w = torch.zeros(D, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    for step in range(steps):
        # Forward pass: prediction and loss
        y_pred = X @ w + b
        loss = torch.mean((y_pred - y) ** 2)
        
        # Backward pass: compute gradients via autograd
        loss.backward()
        
        # Manual gradient descent update (no torch.optim)
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        
        # Zero gradients for next iteration
        w.grad.zero_()
        b.grad.zero_()
    
    # Return detached tensors (no grad tracking)
    return w.detach(), b.detach().squeeze()