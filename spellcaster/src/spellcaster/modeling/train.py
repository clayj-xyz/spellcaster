import torch
from torch import nn

from spellcaster.constants import MODEL_PATH
from .data_loader import build_dataloaders


class BasicConvNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc = nn.Linear(50 * 4 * 4, 128)
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


def train():
    train_dataloader, test_dataloader, classes = build_dataloaders()
    model = BasicConvNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        for x, y in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    model.eval()
    predictions = []
    for x, y in test_dataloader:
        with torch.no_grad():
            y_pred = model(x)
        predictions.extend(
            [(torch.argmax(pred).item(), y_i.item()) for pred, y_i in zip(y_pred, y)]
        )
    
    accuracy = sum([1 for pred, y in predictions if pred == y]) / len(predictions)
    print(f"Accuracy: {accuracy}")


    # Save the model and classes
    combined = {
        "state_dict": model.state_dict(),
        "classes": classes
    }
    torch.save(combined, MODEL_PATH)