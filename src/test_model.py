import torch
from models.cnn import CNN

model = CNN()
test_input = torch.randn(1, 3, 224, 224)

output = model(test_input)

print(output.shape)
