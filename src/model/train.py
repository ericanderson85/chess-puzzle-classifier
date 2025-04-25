from logging import Logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.model.puzzle_cnn import PuzzleCNN
from src.model.puzzle_dataset import LabeledPuzzleDataset, UnlabeledPuzzleDataset, get_datasets
from src.util.config import (
    DATA_SPLIT,
    OPTIMIZATION_FUNCTION,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    DEVICE,
    LOSS_FUNCTION,
    RANDOM_SEED,
    TRAIN_LOG_PATH,
    LEARNING_TYPE,
    LearningType,
    UNSUPERVISED_LOSS_WEIGHT,
    LABEL_CONFIDENCE_THRESHOLD,
)
from src.util.logger import get_logger


def get_dataloaders(labeled_dataset: LabeledPuzzleDataset, unlabeled_dataset: UnlabeledPuzzleDataset, logger: Logger):
    total_labeled_samples = len(labeled_dataset)
    logger.info(f"Total labeled samples: {total_labeled_samples}")

    train_size = int(DATA_SPLIT["train"] * total_labeled_samples)
    if DATA_SPLIT["test"] == 0:
        validate_size = total_labeled_samples - train_size
        test_size = 0
    else:
        validate_size = int(DATA_SPLIT["validate"] * total_labeled_samples)
        test_size = total_labeled_samples - train_size - validate_size

    logger.info(
        f"Splitting labeled data: Train={train_size}, Validate={validate_size}, Test={test_size}")

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_labeled_dataset, validate_dataset, test_dataset = random_split(
        labeled_dataset,
        [train_size, validate_size, test_size],
        generator=generator,
    )

    logger.info(f"Total unlabeled samples: {len(unlabeled_dataset)}")

    labeled_train_loader = DataLoader(train_labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False) if test_size > 0 else None

    unlabeled_train_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return labeled_train_loader, unlabeled_train_loader, validation_loader, test_loader


def train_one_epoch(
    model: PuzzleCNN,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    logger: Logger,
):
    model.train()
    total_labeled_loss = 0.0
    total_unlabeled_loss = 0.0
    correct_labeled = 0
    processed_labeled_samples = 0
    processed_unlabeled_samples = 0
    pseudo_labels_used = 0

    num_unlabeled_batches = len(unlabeled_loader)
    num_labeled_batches = len(labeled_loader)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    steps_per_epoch = max(num_labeled_batches, num_unlabeled_batches)

    for _ in range(steps_per_epoch):
        try:
            labeled_boards, labels = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            labeled_boards, labels = next(labeled_iter)

        labeled_boards = labeled_boards.to(DEVICE)
        labels = labels.to(DEVICE)

        batch_size_labeled = labeled_boards.size(0)
        processed_labeled_samples += batch_size_labeled

        unsupervised_loss = torch.tensor(0.0, device=DEVICE)
        batch_size_unlabeled = 0
        if LEARNING_TYPE == LearningType.SEMI_SUPERVISED and num_unlabeled_batches > 0:
            try:
                unlabeled_boards = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_boards = next(unlabeled_iter)

            unlabeled_boards = unlabeled_boards.to(DEVICE)
            batch_size_unlabeled = unlabeled_boards.size(0)
            processed_unlabeled_samples += batch_size_unlabeled

            model.eval()
            with torch.no_grad():
                unlabeled_outputs_pseudo = model(unlabeled_boards)
                probabilities = torch.softmax(unlabeled_outputs_pseudo, dim=1)
                max_probs, pseudo_labels = torch.max(probabilities, dim=1)
                mask = max_probs >= LABEL_CONFIDENCE_THRESHOLD
            model.train()

            if mask.any():
                confident_boards = unlabeled_boards[mask]
                confident_pseudo_labels = pseudo_labels[mask]
                pseudo_labels_used += confident_pseudo_labels.size(0)

                unlabeled_outputs_train = model(confident_boards)
                unsupervised_loss = loss_fn(
                    unlabeled_outputs_train, confident_pseudo_labels)

        optimizer.zero_grad()

        labeled_outputs = model(labeled_boards)
        supervised_loss = loss_fn(labeled_outputs, labels)

        total_loss = supervised_loss + UNSUPERVISED_LOSS_WEIGHT * unsupervised_loss

        total_loss.backward()
        optimizer.step()

        total_labeled_loss += supervised_loss.item() * batch_size_labeled
        if LEARNING_TYPE == LearningType.SEMI_SUPERVISED:
            total_unlabeled_loss += unsupervised_loss.item() * mask.sum().item()

        predictions = torch.argmax(labeled_outputs, dim=1)
        correct_labeled += (predictions == labels).sum().item()

    avg_labeled_loss = total_labeled_loss / processed_labeled_samples if processed_labeled_samples > 0 else 0
    avg_unlabeled_loss = total_unlabeled_loss / pseudo_labels_used if pseudo_labels_used > 0 else 0
    accuracy = correct_labeled / processed_labeled_samples if processed_labeled_samples > 0 else 0

    return avg_labeled_loss, avg_unlabeled_loss, accuracy, pseudo_labels_used, processed_unlabeled_samples


def validate(model: PuzzleCNN, validation_loader: DataLoader, loss_fn: nn.Module, logger: Logger):
    model.eval()
    total_cost = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for boards, labels in validation_loader:
            boards, labels = boards.to(DEVICE), labels.to(DEVICE)
            outputs = model(boards)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            loss_value = loss_fn(outputs, labels)
            total_cost += loss_value.item() * boards.size(0)
            total_samples += boards.size(0)

    if total_samples == 0:
        logger.warning("Validation set is empty.")
        return 0.0, 0.0

    average_cost = total_cost / total_samples
    accuracy = correct / total_samples
    return average_cost, accuracy


def gradient_descent(
    model: PuzzleCNN,
    labeled_train_loader: DataLoader,
    unlabeled_train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    logger: Logger,
):
    validation_loss, validation_accuracy = validate(
        model, validation_loader, loss_fn, logger)
    logger.info(
        f"Epoch [0/{NUM_EPOCHS}] | "
        f"Validation Loss: {validation_loss:.4f} | "
        f"Validation Accuracy: {(validation_accuracy * 100):.2f}%"
    )

    best_validation_accuracy = 0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        (
            train_labeled_loss,
            train_unlabeled_loss,
            train_accuracy,
            pseudo_labels_used,
            unlabeled_processed,
        ) = train_one_epoch(
            model,
            labeled_train_loader,
            unlabeled_train_loader,
            optimizer,
            loss_fn,
            logger,
        )

        validation_loss, validation_accuracy = validate(
            model, validation_loader, loss_fn, logger
        )

        if validation_accuracy > best_validation_accuracy:
            best_epoch = epoch
            best_validation_accuracy = validation_accuracy

        log_msg = (
            f"Epoch [{epoch}/{NUM_EPOCHS}]\n"
            f"\tTrain Labeled Loss: {train_labeled_loss:.4f}\n"
            f"\tTrain Accuracy: {(train_accuracy * 100):.2f}%\n"
            f"\tValidation Loss: {validation_loss:.4f}\n"
            f"\tValidation Accuracy: {(validation_accuracy * 100):.2f}%\n"
        )
        if LEARNING_TYPE == LearningType.SEMI_SUPERVISED:
            unlab_fraction = pseudo_labels_used / unlabeled_processed if unlabeled_processed > 0 else 0
            log_msg += (
                f"\tTrain Unlabeled Loss: {train_unlabeled_loss:.4f}\n"
                f"\tPseudo Labels Used: {pseudo_labels_used}/{unlabeled_processed} ({unlab_fraction*100:.1f}%)\n\n"
            )
        logger.info(log_msg)

    logger.info(
        f"Best Epoch: {best_epoch}\n"
        f"\tValidation Accuracy: {(best_validation_accuracy*100):.2f}%"
    )
    if test_loader and len(test_loader.dataset) > 0:
        test_loss, test_accuracy = validate(model, test_loader, loss_fn, logger)
        logger.info(
            f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {(test_accuracy*100):.2f}%")


def main():
    logger = get_logger(__name__, TRAIN_LOG_PATH)
    logger.info(
        f"Starting training process with LEARNING_TYPE = {LEARNING_TYPE.name}")
    logger.info(f"Using device: {DEVICE}")

    labeled_dataset, unlabeled_dataset = get_datasets(logger)
    labeled_train_loader, unlabeled_train_loader, validation_loader, test_loader = get_dataloaders(
        labeled_dataset, unlabeled_dataset, logger)

    model = PuzzleCNN().to(DEVICE)
    optimizer = OPTIMIZATION_FUNCTION(model.parameters(), lr=LEARNING_RATE)
    loss_fn = LOSS_FUNCTION()

    gradient_descent(
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
