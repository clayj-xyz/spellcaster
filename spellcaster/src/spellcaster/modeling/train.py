import torch
from easyfsl.utils import sliding_average
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from spellcaster.constants import MODEL_PATH
from .data import omniglot_dataloaders, spell_dataloader
from .net import FewShotClassifier


def evaluate_on_one_task(
    model: torch.nn.Module,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> tuple[int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    return (
        torch.max(
            model(support_images, support_labels, query_images)
            .detach()
            .data,
            1,
        )[1]
        == query_labels
    ).sum().item(), len(query_labels)


def evaluate(model: torch.nn.Module, data_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval()
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):

            correct, total = evaluate_on_one_task(
                model, support_images, support_labels, query_images, query_labels
            )

            total_predictions += total
            correct_predictions += correct

    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
    )


def train():
    train_loader, test_loader = omniglot_dataloaders()
    model = FewShotClassifier()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    log_update_frequency = 10

    all_loss = []
    model.train()
    with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in tqdm_train:
            optimizer.zero_grad()
            classification_scores = model(
                support_images, support_labels, query_images
            )

            loss = criterion(classification_scores, query_labels)
            all_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            if episode_index % log_update_frequency == 0:
                tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))

    evaluate(model, test_loader)
    torch.save(model.state_dict(), MODEL_PATH)


def evaluate_on_spell_classification():
    spell_loader = spell_dataloader()
    model = FewShotClassifier()
    model.load_state_dict(torch.load(MODEL_PATH))
    evaluate(model, spell_loader)