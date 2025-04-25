from logging import Logger
import os
import chess
from chess import Board, pgn, WHITE
import torch
from torch.utils.data import Dataset

from src.util.config import FEW_PUZZLE_COUNT, LEARNING_TYPE, PUZZLE_CLASSES, PUZZLE_REPRESENTATION, PUZZLES_DIRECTORY, TRAIN_LOG_PATH, BoardRepresentation, BOARD_REPRESENTATION, PIECE_INDICES_MAP, LearningType, PuzzleRepresentation
from src.util.chess_util import get_game


def get_board_tensor(board: Board) -> torch.Tensor:
    """
    Depending on BOARD_REPRESENTATION, the output will be:
      - PIECE_INDEX (each square gets the piece's index, signed for color)
      - SIGNED_ONE_HOT (6 channels, one per piece type with sign for color)
      - ONE_HOT (12 channels, separated for white and black)
    """
    channels, height, width = BOARD_REPRESENTATION.value

    if BOARD_REPRESENTATION == BoardRepresentation.PIECE_INDEX:
        board_matrix = [[0 for _ in range(width)] for _ in range(height)]
        for file in range(8):
            for rank in range(8):
                square = chess.square(file, rank)
                piece_type = board.piece_at(square)
                if piece_type is None:
                    continue
                piece_index = PIECE_INDICES_MAP[piece_type.symbol()]
                board_matrix[rank][file] = piece_index

        board_tensor = torch.tensor(board_matrix, dtype=torch.float32)
        board_tensor = board_tensor.unsqueeze(0)

    elif BOARD_REPRESENTATION == BoardRepresentation.SIGNED_ONE_HOT:
        board_matrix = [[[0 for _ in range(6)] for _ in range(width)]
                        for _ in range(height)]
        for file in range(8):
            for rank in range(8):
                square = chess.square(file, rank)
                piece_type = board.piece_type_at(square)
                if piece_type is None:
                    continue
                color = board.color_at(square)
                board_matrix[rank][file][piece_type - 1] = 1 if color else -1
        board_tensor = torch.tensor(board_matrix, dtype=torch.float32)
        board_tensor = board_tensor.permute(2, 0, 1)

    else:
        board_matrix = [
            [[0 for _ in range(12)] for _ in range(width)] for _ in range(height)]
        for file in range(8):
            for rank in range(8):
                square = chess.square(file, rank)
                piece_type = board.piece_type_at(square)
                if piece_type is None:
                    continue
                color = board.color_at(square)
                if color == WHITE:
                    board_matrix[rank][file][piece_type - 1] = 1
                else:
                    board_matrix[rank][file][piece_type - 1 + 6] = 1

        board_tensor = torch.tensor(board_matrix, dtype=torch.float32)
        board_tensor = board_tensor.permute(2, 0, 1)

    return board_tensor


def get_puzzle_tensor_from_game(game: pgn.Game) -> torch.Tensor:
    """""
    Depending on PUZZLE_REPRESENTATION, the output will be:
      - FIRST_ONLY (only the starting board)
      - FIRST_AND_LAST (concatenates starting and final boards)
      - ALL_POSITIONS (uses a fixed number of positions)
    """

    fen = game.headers.get("FEN")
    if fen:
        board = Board(fen)
    else:
        raise ValueError('PGN missing FEN header')

    boards = [get_board_tensor(board)]

    node = game
    while node.variations:
        move = node.variation(0).move
        board.push(move)
        boards.append(get_board_tensor(board))
        node = node.variation(0)

    if PUZZLE_REPRESENTATION == PuzzleRepresentation.FIRST:
        selected = boards[0:1]

    elif PUZZLE_REPRESENTATION == PuzzleRepresentation.FIRST_AND_LAST:
        selected = [boards[0], boards[-1]]

    else:
        n = len(boards)
        if n < FEW_PUZZLE_COUNT:
            # Pad the list with zeros
            zeros_tensor = torch.zeros_like(boards[0])
            selected = boards + [zeros_tensor] * (FEW_PUZZLE_COUNT - n)
        elif n > FEW_PUZZLE_COUNT:
            # Sample evenly from the puzzle, including first and last
            indices = [
                int(round(i * (n - 1) / (FEW_PUZZLE_COUNT - 1)))
                for i in range(FEW_PUZZLE_COUNT)
            ]
            selected = [boards[i] for i in indices]
        else:
            selected = boards

    puzzle_tensor = torch.cat(selected, dim=0)
    return puzzle_tensor


def load_puzzles(logger: Logger) -> tuple[list[torch.Tensor], list[int], list[torch.Tensor]]:
    labeled_puzzles = []
    labeled_puzzle_ids = []
    labels = []
    unlabeled_puzzles = []
    unlabeled_puzzle_ids = []

    puzzle_ids = os.listdir(PUZZLES_DIRECTORY)
    puzzle_ids = filter(lambda file: file.endswith(".pgn"), puzzle_ids)
    puzzle_ids = list(map(lambda file: int(file[:-4]), puzzle_ids))
    puzzle_ids.sort()

    labeled_count = 0
    unlabeled_count = 0

    for puzzle_id in puzzle_ids:
        pgn_path = os.path.join(PUZZLES_DIRECTORY, f"{puzzle_id}.pgn")

        try:
            game = get_game(pgn_path, logger)

            puzzle_tensor = get_puzzle_tensor_from_game(game)
            if puzzle_tensor is None:
                logger.warning(f"Could not create tensor for {puzzle_id}")
                continue

            label_string = game.headers.get("Label", None)

            if label_string and label_string in PUZZLE_CLASSES:
                labeled_puzzles.append(puzzle_tensor)
                labeled_puzzle_ids.append(puzzle_id)
                label_id = PUZZLE_CLASSES.index(label_string)
                labels.append(label_id)
                labeled_count += 1
            else:
                unlabeled_puzzles.append(puzzle_tensor)
                unlabeled_puzzle_ids.append(puzzle_id)
                unlabeled_count += 1

        except Exception as e:
            logger.error(f"Error processing {puzzle_id}: {e}")

    logger.info(
        f"Processed {len(puzzle_ids)} files. Labeled: {labeled_count}, Unlabeled: {unlabeled_count}")

    return labeled_puzzles, labeled_puzzle_ids, labels, unlabeled_puzzles, unlabeled_puzzle_ids


class LabeledPuzzleDataset(Dataset):
    def __init__(self, labeled_puzzles: list[torch.Tensor], labeled_puzzle_ids: list[int], labels: list[int]) -> None:
        self.puzzles = labeled_puzzles
        self.puzzle_ids = labeled_puzzle_ids
        self.labels = labels
        if len(labeled_puzzles) != len(labels):
            raise ValueError("Puzzles and Labels must have the same length.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        return self.puzzles[idx], self.labels[idx]

    def get_puzzle_id(self, idx) -> int:
        return self.puzzle_ids[idx]


class UnlabeledPuzzleDataset(Dataset):
    def __init__(self, unlabeled_puzzles: list[torch.Tensor], unlabeled_puzzle_ids: list[int],) -> None:
        self.puzzles = unlabeled_puzzles
        self.puzzle_ids = unlabeled_puzzle_ids

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.puzzles[idx]

    def get_puzzle_id(self, idx) -> int:
        return self.puzzle_ids[idx]


def get_datasets(logger: Logger) -> tuple[LabeledPuzzleDataset, UnlabeledPuzzleDataset]:
    labeled_puzzles, labeled_puzzle_ids, labels, unlabeled_puzzles, unlabeled_puzzle_ids = load_puzzles(
        logger)
    labled_dataset = LabeledPuzzleDataset(labeled_puzzles, labeled_puzzle_ids, labels)
    unlabled_dataset = UnlabeledPuzzleDataset(unlabeled_puzzles, unlabeled_puzzle_ids)
    return labled_dataset, unlabled_dataset
