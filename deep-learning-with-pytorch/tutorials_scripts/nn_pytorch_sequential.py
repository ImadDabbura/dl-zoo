"""
Fitting a random data using fully connected network with one hidden layer.
"""


import torch


device = torch.device('gpu:0' if torch.cuda.is_available() else 'cpu')

# Define dimensions
N, D_IN, H, D_OUT = 64, 1000, 100, 10

# Create random data
X = torch.randn(N, D_IN, device=device, dtype=dtype, requires_grad=False)
y = torch.randn(N, D_OUT, device=device, dtype=dtype, requires_grad=False)

# Defone a sequential model
model = torch.nn.Sequential(
    torch.nn.Linear(D_IN, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_OUT)
)

# Define loss function
criterion = torch.nn.MSELoss(size_average=False)

# Define learning rate
lr = 1e-4

# Loop over number of epochs to fit the data to nn
for epoch in range(50):
    # Forward pass
    y_pred = model(X)

    # Compute loss
    loss = criterion(y_pred, y)
    print(f'Loss {epoch} : {loss.item():.4f}')

    # Use autograd to compute gradients
    model.zero_grad()

    # Compute gradients
    loss.backward()

    # Update wights using gradient descent
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad
