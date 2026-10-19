# import torch
# import torch.nn as nn
# import torch.optim as optim

# # 1. Training data
# X = torch.tensor([[2.0], [3.0], [4.0], [5.0], [6.0]])
# y = torch.tensor([[4.0], [6.0], [8.0], [10.0], [12.0]])

# # 2. Model: a single linear neuron y = Wx + b
# model = nn.Linear(1, 1)

# # 3. Loss + optimizer
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# # 4. Training loop
# for epoch in range(200):
#     optimizer.zero_grad()      # Clear old gradients
#     y_pred = model(X)          # Forward pass
#     loss = criterion(y_pred, y) # Compute loss
#     loss.backward()            # Compute gradients
#     optimizer.step()           # Update weights

#     if epoch % 20 == 0:
#         print(epoch, loss.item())

# # 5. Final learned parameters
# print("W =", model.weight.item())
# print("b =", model.bias.item())

# without using pytorch

# Training data
X = [2, 3, 4, 5, 6]
Y = [4, 6, 8, 10, 12]   # y = 2x

# 1. Initialize weights manually (random small values)
W = 0.5     # try starting at 0.5
B = 0.0     # bias starts at 0

# 2. Hyperparameters
lr = 0.01   # learning rate
epochs = 20

# 3. Training loop
for epoch in range(epochs):
    total_loss = 0

    # Loop through each training example
    for x, y in zip(X, Y):

        # ---- Forward pass ----
        y_pred = W * x + B

        # ---- Loss (MSE) ----
        loss = (y_pred - y) ** 2
        total_loss += loss

        # ---- Compute gradients manually ----
        # dLoss/dW = 2*(y_pred - y)*x
        dW = 2 * (y_pred - y) * x

        # dLoss/dB = 2*(y_pred - y)
        dB = 2 * (y_pred - y)

        # ---- Update weights ----
        W = W - lr * dW
        B = B - lr * dB

    print(f"Epoch {epoch}: Loss={total_loss:.4f}, W={W:.4f}, B={B:.4f}")
