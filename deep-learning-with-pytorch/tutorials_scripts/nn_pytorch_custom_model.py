"""
Fitting a random data using fully connected network with one hidden layer.
"""


import torch


class Net(torch.nn.Module):
    """Implementing two layer nn."""

    def __init__(self, D_IN, H, D_OUT):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_IN, H)
        self.linear2 = torch.nn.Linear(H, D_OUT)

    def forward(self, x):
        h = self.linear1(x)
        h_relu = torch.clamp(h, 0)
        y_pred = self.linear2(h_relu)
        return y_pred


dtype = torch.float
device = torch.device('gpu:0' if torch.cuda.is_available() else 'cpu')

# Define dimensions
N, D_IN, H, D_OUT = 64, 1000, 100, 10

# Create random data
X = torch.randn(N, D_IN, device=device, dtype=dtype, requires_grad=False)
y = torch.randn(N, D_OUT, device=device, dtype=dtype, requires_grad=False)

# Instantiate the model
model = net(D_IN, H, D_OUT)

# Define loss function
criterion = torch.nn.MSELoss(size_average=False)

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# Loop over number of epochs to fit the data to nn
for epoch in range(50):
    # Forward pass
    y_pred = model(X)

    # Compute loss
    loss = criterion(y_pred, y)
    print(f'Loss {epoch} : {loss.item():.4f}')

    # Use autograd to compute gradients
    optimizer.zero_grad()

    # Compute gradients
    loss.backward()

    # Update wights using gradient descent
    optimizer.step()
