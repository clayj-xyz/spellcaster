import os

import numpy as np
from easyfsl.samplers import TaskSampler
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, Omniglot
from torchvision import transforms

from spellcaster.constants import MODEL_INPUT_SIZE, DATA_DIR


def omniglot_dataloaders(
    nway=5,
    nshot=5,
    nquery=10,
    ntraining_tasks=40_000,
    nevaluation_tasks=100,
    num_workers=12,
):
    pretrain_data_dir = os.path.join(DATA_DIR, "pretrain")

    # background=True selects the train set, background=False selects the test set
    train_set = Omniglot(
        root=pretrain_data_dir,
        background=True,
        transform=transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.RandomResizedCrop(MODEL_INPUT_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )
    test_set = Omniglot(
        root=pretrain_data_dir,
        background=False,
        transform=transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(
                    [int(MODEL_INPUT_SIZE * 1.15), int(MODEL_INPUT_SIZE * 1.15)]
                ),
                transforms.CenterCrop(MODEL_INPUT_SIZE),
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )

    train_set.get_labels = lambda: [instance[1] for instance in train_set._flat_character_images]
    train_sampler = TaskSampler(
        train_set, n_way=nway, n_shot=nshot, n_query=nquery, n_tasks=ntraining_tasks
    )
    test_set.get_labels = lambda: [
        instance[1] for instance in test_set._flat_character_images
    ]
    test_sampler = TaskSampler(
        test_set, n_way=nway, n_shot=nshot, n_query=nquery, n_tasks=nevaluation_tasks
    )

    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    return train_loader, test_loader


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
    

class IncreaseContrast():
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def __call__(self, img):
        img[img < self.threshold] = 0
        img[img >= self.threshold] = 1
        return img
    

def spell_dataloader():
    image_dir = os.path.join(DATA_DIR, "images")
    dataset = ImageFolder(image_dir, transform=transforms.Compose(
        [
        transforms.Grayscale(),
        CropWandPath(),
        transforms.Lambda(
            lambda x: 1 - (torch.from_numpy(x).to(torch.float32).unsqueeze(0) / 255)
        ),
        transforms.Lambda(
            lambda x: torch.nn.functional.interpolate(
                x.unsqueeze(0), size=(28, 28), mode="bilinear", antialias=True
            ).squeeze(0)
        ),
        IncreaseContrast(),
        ]
    ))

    dataset.get_labels = lambda: dataset.targets
    sampler = TaskSampler(
        dataset, n_way=len(dataset.classes), n_shot=5, n_query=5, n_tasks=100
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=sampler.episodic_collate_fn,
    )
