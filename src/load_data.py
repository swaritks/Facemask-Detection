from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Make all images 224x224
    transforms.ToTensor()
])

dataset = ImageFolder(root="data", transform=transform)                 # Data is found within "data" folder, transform converts each image
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)           # Load images in groups of 32 to increase GPU efficiency

images, labels = next(iter(dataloader))                                 # next(iter(dataloader)) grabs the first batch of images
print(images.shape, labels.shape)                                       # ensuring size and shape are consistent