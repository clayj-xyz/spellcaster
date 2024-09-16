import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

from spellcaster.constants import DATA_DIR, MODEL_PATH


class BasicConvNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc = nn.Linear(50 * 13 * 13, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc(x))
        x = self.classifier(x)
        return x

class CropWandPath():
    def __init__(self, padding=10):
        self.padding = padding

    def __call__(self, img):
        img = np.array(img)
        x, y = np.nonzero(img)
        xl,xr = x.min(),x.max()
        yl,yr = y.min(),y.max()
        largest_side = max(xr - xl, yr - yl)
        return img[xl:xl+largest_side, yl:yl+largest_side]


def train():
    image_dir = os.path.join(DATA_DIR, "images")
    dataset = ImageFolder(image_dir, transform=transforms.Compose(
        [
        transforms.Grayscale(),
        CropWandPath(),
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        ]
    ))
    classes = dataset.classes

    # apply train test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Load the model
    model = BasicConvNet()

    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    model.eval()
    predictions = []
    for x, y in test_dataset:
        with torch.no_grad():
            y_pred = model(x.unsqueeze(0))
        predictions.append((y_pred.argmax(), y))
    
    accuracy = sum([1 for pred, y in predictions if pred == y]) / len(predictions)
    print(f"Accuracy: {accuracy}")


    # Save the model and classes
    combined = {
        "state_dict": model.state_dict(),
        "classes": classes
    }
    torch.save(combined, MODEL_PATH)