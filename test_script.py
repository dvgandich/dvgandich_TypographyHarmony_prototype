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

# Подсчёт параметров вручную
total_params = 0
for name, param in model.named_parameters():
    params_in_layer = param.numel()
    total_params += params_in_layer
    print(f"{name}: {params_in_layer} params")

print(f"Total parameters: {total_params}")

# Ожидаемое число параметров:
# Linear(11,32): 11*32 + 32 = 352 + 32 = 384? Нет, правильно: 11*32 + 32 = 352 + 32 = 384
# Linear(32,16): 32*16 + 16 = 512 + 16 = 528
# Linear(16,8): 16*8 + 8 = 128 + 8 = 136
# Linear(8,1): 8*1 + 1 = 8 + 1 = 9
# Итого: 384 + 528 + 136 + 9 = 1057

# Обновляем ожидаемое значение
expected_params = 1057
assert total_params == expected_params, f"Expected {expected_params} params, got {total_params}"
print(f"Neural network created. Parameters: {total_params}")
print("All tests passed!")
