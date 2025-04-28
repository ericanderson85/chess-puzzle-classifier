import torch.nn.functional as F
import numpy as np
from logging import Logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from src.model.puzzle_cnn import PuzzleCNN
from src.model.puzzle_dataset import LabeledPuzzleDataset, UnlabeledPuzzleDataset, get_datasets
from src.util.config import Config, LearningType
from src.util.logger import get_logger
from src.util.plotting import line_plot


def get_dataloaders(
    config: Config,
    labeled_dataset: LabeledPuzzleDataset,
    unlabeled_dataset: UnlabeledPuzzleDataset,
    logger: Logger,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:

    total_labeled = len(labeled_dataset)
    total_unlabeled = len(unlabeled_dataset)
    logger.info(f"Total labeled samples: {total_labeled}")
    logger.info(f"Total unlabeled samples: {total_unlabeled}")

    train_ratio = config.DATA_SPLIT["train"]
    val_ratio = config.DATA_SPLIT["validate"]
    train_prop = int(train_ratio * total_labeled)
    val_prop = int(val_ratio * total_labeled)
    test_prop = total_labeled - train_prop - val_prop

    logger.info(
        f"Splitting labeled data: "
        f"Train={train_prop}, Validate={val_prop}, Test={test_prop}"
    )

    g = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_ds_full, val_ds, test_ds = random_split(
        labeled_dataset,
        [train_prop, val_prop, test_prop],
        generator=g,
    )

    if (
        config.NUM_SAMPLES is not None
        and config.NUM_SAMPLES < len(train_ds_full)
    ):
        limited_idxs = train_ds_full.indices[: config.NUM_SAMPLES]
        train_ds = Subset(labeled_dataset, limited_idxs)
        logger.info(f"Capping train samples to {config.NUM_SAMPLES}")
    else:
        train_ds = train_ds_full

    if (
        config.NUM_UNLABELED_SAMPLES is not None
        and config.NUM_UNLABELED_SAMPLES < total_unlabeled
    ):
        unlabeled_idxs = torch.randperm(
            total_unlabeled, generator=g
        )[: config.NUM_UNLABELED_SAMPLES]
        unlabeled_ds = Subset(unlabeled_dataset, unlabeled_idxs.tolist())
        logger.info(
            f"Capping unlabeled samples to {config.NUM_UNLABELED_SAMPLES}"
        )
    else:
        unlabeled_ds = unlabeled_dataset

    labeled_train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True
    )
    validation_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False
    )
    test_loader = (
        DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        if test_prop > 0
        else None
    )
    unlabeled_train_loader = DataLoader(
        unlabeled_ds, batch_size=config.BATCH_SIZE, shuffle=True
    )

    return (
        labeled_train_loader,
        unlabeled_train_loader,
        validation_loader,
        test_loader,
    )


def train_one_epoch(
    config: Config,
    model: PuzzleCNN,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    logger: Logger,
) -> tuple[float, float, float, int, int]:
    model.train()

    total_loss = 0.0
    total_sup_loss = 0.0
    total_unsup_loss = 0.0
    correct_labeled = 0
    processed_labeled = 0
    processed_unlabeled = 0
    pseudo_labels_used = 0

    unlabeled_iter = iter(unlabeled_loader)

    for boards, labels in labeled_loader:
        boards = boards.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        batch_size = boards.size(0)
        processed_labeled += batch_size

        try:
            unlabeled_boards = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unlabeled_boards = next(unlabeled_iter)

        unlabeled_boards = unlabeled_boards.to(config.DEVICE)
        processed_unlabeled += unlabeled_boards.size(0)

        outputs = model(boards)
        sup_loss = loss_fn(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        correct_labeled += (preds == labels).sum().item()

        loss = sup_loss
        total_sup_loss += sup_loss.item() * batch_size

        if config.LEARNING_TYPE == LearningType.SEMI_SUPERVISED:
            with torch.no_grad():
                unlabeled_outputs = model(unlabeled_boards)
                prob = torch.softmax(unlabeled_outputs, dim=1)
                max_probs, pseudo_labels = torch.max(prob, dim=1)
                mask = max_probs >= config.LABEL_CONFIDENCE_THRESHOLD

            if mask.sum() > 0:
                confident_unlabeled = unlabeled_boards[mask]
                confident_pseudo_labels = pseudo_labels[mask]
                pseudo_labels_used += mask.sum().item()

                confident_outputs = model(confident_unlabeled)
                unsup_loss = loss_fn(confident_outputs, confident_pseudo_labels)

                weighted_unsup_loss = config.UNSUPERVISED_LOSS_WEIGHT * unsup_loss
                loss += weighted_unsup_loss
                total_unsup_loss += unsup_loss.item() * mask.sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size

    avg_sup_loss = total_sup_loss / processed_labeled if processed_labeled > 0 else 0.0
    avg_unsup_loss = total_unsup_loss / pseudo_labels_used if pseudo_labels_used > 0 else 0.0
    accuracy = correct_labeled / processed_labeled if processed_labeled > 0 else 0.0

    return (
        avg_sup_loss,
        avg_unsup_loss,
        accuracy,
        pseudo_labels_used,
        processed_unlabeled,
    )


def validate(
    config: Config,
    model: PuzzleCNN,
    validation_loader: DataLoader,
    loss_fn: nn.Module,
    logger: Logger
):

    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for boards, labels in validation_loader:
            boards, labels = boards.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(boards)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            loss_value = loss_fn(outputs, labels)
            total_loss += loss_value.item() * boards.size(0)
            total_samples += boards.size(0)

    if total_samples == 0:
        logger.warning("Validation set is empty.")
        return 0.0, 0.0

    average_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return average_loss, accuracy


def gradient_descent(
    config: Config,
    model: PuzzleCNN,
    labeled_train_loader: DataLoader,
    unlabeled_train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    logger: Logger,
) -> tuple[float, float]:

    epochs = []
    val_accuracies = []
    train_accuracies = []

    validation_loss, validation_accuracy = validate(
        config, model, validation_loader, loss_fn, logger)

    epochs.append(0)
    val_accuracies.append(validation_accuracy)
    train_accuracies.append(0)

    logger.info(
        f"Epoch 0 | "
        f"Validation Loss: {validation_loss:.4f} | "
        f"Validation Accuracy: {(validation_accuracy * 100):.2f}%"
    )

    best_validation_loss = 1_000_000
    best_train_loss = 1_000_000
    best_validation_accuracy = 0
    best_epoch = 0

    no_improve = 0

    for epoch in range(1, config.MAX_EPOCHS + 1):
        if no_improve >= config.EARLY_STOPPING_PATIENCE:
            break

        (
            train_labeled_loss,
            train_unlabeled_loss,
            train_accuracy,
            pseudo_labels_used,
            unlabeled_processed,
        ) = train_one_epoch(
            config,
            model,
            labeled_train_loader,
            unlabeled_train_loader,
            optimizer,
            loss_fn,
            logger,
        )

        validation_loss, validation_accuracy = validate(
            config, model, validation_loader, loss_fn, logger
        )

        epochs.append(epoch)
        val_accuracies.append(validation_accuracy)
        train_accuracies.append(train_accuracy)

        if validation_loss < best_validation_loss - config.EARLY_STOPPING_DELTA:
            best_validation_loss = validation_loss
            best_train_loss = train_labeled_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        log_msg = (
            f"Epoch {epoch}\n"
            f"\tTrain Labeled Loss: {train_labeled_loss:.4f}\n"
            f"\tTrain Accuracy: {(train_accuracy * 100):.2f}%\n"
            f"\tValidation Loss: {validation_loss:.4f}\n"
            f"\tValidation Accuracy: {(validation_accuracy * 100):.2f}%\n"
        )
        if config.LEARNING_TYPE == LearningType.SEMI_SUPERVISED:
            unlab_fraction = pseudo_labels_used / unlabeled_processed if unlabeled_processed > 0 else 0
            log_msg += (
                f"\tTrain Unlabeled Loss: {train_unlabeled_loss:.4f}\n"
                f"\tPseudo Labels Used: {pseudo_labels_used}/{unlabeled_processed} ({unlab_fraction*100:.1f}%)\n\n"
            )
        logger.info(log_msg)

    logger.info(
        f"Best Epoch: {best_epoch}\n"
        f"\tValidation Loss: {(best_validation_loss):.4f}%"
        f"\tValidation Accuracy: {(best_validation_accuracy*100):.2f}%"
    )

    line_plot(
        x=np.array(epochs),
        y=[np.array(train_accuracies), np.array(val_accuracies)],
        logger=logger,
        title="Model Accuracy Over Training",
        xlabel="Epoch",
        ylabel="Accuracy",
        labels=["Training Accuracy", "Validation Accuracy"],
        filename="accuracy_plot.png"
    )

    if test_loader and len(test_loader.dataset) > 0:
        test_loss, test_accuracy = validate(config, model, test_loader, loss_fn, logger)
        logger.info(
            f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {(test_accuracy*100):.2f}%")

    return best_train_loss, best_validation_loss


def main():
    config = Config()
    logger = get_logger(__name__, config.TRAIN_LOG_PATH)
    logger.info(
        f"Starting training process with config.LEARNING_TYPE = {config.LEARNING_TYPE.name}")
    logger.info(f"Using device: {config.DEVICE}")

    labeled_dataset, unlabeled_dataset = get_datasets(config, logger)
    labeled_train_loader, unlabeled_train_loader, validation_loader, test_loader = get_dataloaders(
        config, labeled_dataset, unlabeled_dataset, logger)

    model = PuzzleCNN(config).to(config.DEVICE)
    optimizer = config.OPTIMIZATION_FUNCTION(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = config.LOSS_FUNCTION()

    gradient_descent(
        config,
        model,
        labeled_train_loader,
        unlabeled_train_loader,
        validation_loader,
        test_loader,
        optimizer,
        loss_fn,
        logger,
    )

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
