import torch
from models.cnn import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)