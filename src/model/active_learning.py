import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch import nn
from src.dataset.classify_puzzles import classify_puzzle
from src.model.train import get_dataloaders, gradient_descent
from src.model.puzzle_cnn import PuzzleCNN
from src.model.puzzle_dataset import UnlabeledPuzzleDataset, get_datasets
from src.util.config import ACTIVE_LEARNING_LOG_PATH, DEVICE, BATCH_SIZE, LEARNING_RATE, LOSS_FUNCTION, OPTIMIZATION_FUNCTION, PUZZLES_DIRECTORY
from src.util.logger import get_logger
from src.util.chess_util import get_game, write_game_to_file


def get_uncertainty_scores(model: PuzzleCNN, unlabeled_loader: DataLoader):
    model.eval()
    uncertainties = []
    indices = []

    with torch.no_grad():
        for boards, idx in unlabeled_loader:
            boards = boards.to(DEVICE)
            outputs = model(boards)
            probabilities = torch.softmax(outputs, dim=1)

            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)

            uncertainties.extend(entropy.cpu().numpy())
            indices.extend(idx.cpu().numpy())

    return np.array(uncertainties), np.array(indices)


def select_samples_for_labeling(model: PuzzleCNN, unlabeled_dataset: UnlabeledPuzzleDataset, num_samples: int = 10) -> tuple[list[int], ]:

    def collate_fn(batch):
        boards, indices = zip(*batch)
        return torch.stack(boards), torch.tensor(indices)

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    uncertainty_scores, indices = get_uncertainty_scores(model, unlabeled_loader)
    most_uncertain_idxs = np.argsort(uncertainty_scores)[-num_samples:]
    selected_indices = indices[most_uncertain_idxs]
    selected_puzzle_ids = [unlabeled_dataset.get_puzzle_id(idx) for idx in selected_indices]

    return selected_puzzle_ids


def active_learning_loop(num_iterations=5, samples_per_iteration=10):
    logger = get_logger(__name__, ACTIVE_LEARNING_LOG_PATH)
    logger.info("Starting active learning process")

    for iteration in range(num_iterations):
        logger.info(f"Active Learning Iteration {iteration+1}/{num_iterations}")

        model = PuzzleCNN().to(DEVICE)
        labeled_dataset, unlabeled_dataset = get_datasets(logger)
        labeled_train_loader, unlabeled_train_loader, validation_loader, test_loader = get_dataloaders(
            labeled_dataset, unlabeled_dataset, logger)
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
            is_semi_supervised=False
        )

        selected_puzzle_ids = select_samples_for_labeling(
            model, unlabeled_dataset, samples_per_iteration)

        newly_labeled = []
        for puzzle_id in selected_puzzle_ids:
            puzzle_path = os.path.join(PUZZLES_DIRECTORY, f'{puzzle_id}.pgn')
            game = get_game(puzzle_path, logger)

            if game is None:
                continue

            print(f"\n--- Puzzle ID: {puzzle_id} ---")

            label = classify_puzzle(game, logger)

            if label is None:
                continue

            if label == 'rem':
                os.remove(puzzle_path)

            game.headers['Label'] = label
            write_game_to_file(game, puzzle_path, logger)
            newly_labeled.append(puzzle_id)
            logger.info(f"Labeled puzzle {puzzle_id} as {label}")

        logger.info(f"Labeled {len(newly_labeled)} new puzzles in this iteration")


if __name__ == '__main__':
    active_learning_loop()
