import os
import numpy as np

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

from spellcaster.constants import DATA_DIR


class CropWandPath():
    def __init__(self, padding=5):
        self.padding = padding

    def __call__(self, img):
        img = np.array(img)
        x, y = np.nonzero(img)
        xl, xr = x.min(), x.max()
        yl, yr = y.min(), y.max()
        largest_side = max(xr - xl, yr - yl)
        crop_size = largest_side + 2 * self.padding
        
        # Calculate the center of the nonzero area
        center_x = (xl + xr) // 2
        center_y = (yl + yr) // 2
        
        # Calculate the new bounds to center the nonzero area
        new_xl = max(0, center_x - crop_size // 2)
        new_xr = new_xl + crop_size
        new_yl = max(0, center_y - crop_size // 2)
        new_yr = new_yl + crop_size
        
        return img[new_xl:new_xr, new_yl:new_yr]


def build_dataloaders(batch_size=16):
    image_dir = os.path.join(DATA_DIR, "images")
    dataset = ImageFolder(image_dir, transform=transforms.Compose(
        [
        transforms.Grayscale(),
        CropWandPath(),
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        ]
    ))

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader, dataset.classes