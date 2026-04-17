import numpy as np
import torch
import torch.nn as nn

print("All imports OK")

def harmony_function(g, h_target, sigma):
    if h_target <= 0:
        h_target = 0.01
    if sigma <= 0:
        sigma = 0.01
    deviation = (g / h_target) - 1
    return np.exp(-(deviation ** 2) / (2 * sigma ** 2))

result = harmony_function(1.3, 1.3, 0.2)
assert 0.9 < result <= 1.0
print(f"Harmony function test passed: {result}")

class HarmonyNN(nn.Module):
    def __init__(self, input_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = HarmonyNN()
params = sum(p.numel() for p in model.parameters())
print(f"Neural network created. Parameters: {params}")
assert params == 1025
print("All tests passed!")
