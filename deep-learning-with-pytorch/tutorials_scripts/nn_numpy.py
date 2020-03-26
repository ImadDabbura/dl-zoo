"""
Fitting a random data using fully connected network with one hidden layer.
"""


import numpy as np


# Define dimensions
N, D_IN, H, D_OUT = 64, 1000, 100, 10

# Create random data
X = np.random.randn(N, D_IN)
y = np.random.randn(N, D_OUT)

# Initialize weights randomly
w_1 = np.random.randn(D_IN, H)
w_2 = np.random.randn(H, D_OUT)

# Define learning rate
lr = 1e-6

# Loop over number of epochs to fit the data to nn
for epoch in range(50):
    # Forward pass
    h = X.dot(w_1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w_2)

    # Compute loss
    loss = np.square(y_pred - y).sum()
    print(f'Loss {epoch} : {loss:.4f}')

    # Backward pass
    y_pred_grad = 2 * (y_pred - y)
    w_2_grad = h_relu.T.dot(y_pred_grad)
    h_relu_grad = y_pred_grad.dot(w_2.T)
    h_grad = h_relu_grad.copy()
    h_grad[h < 0] = 0
    w_1_grad = X.T.dot(h_grad)

    # Update wights using gradient descent
    w_1 -= lr * w_1_grad
    w_2 -= lr * w_2_grad
