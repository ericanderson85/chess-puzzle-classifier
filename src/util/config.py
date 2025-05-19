from dataclasses import dataclass, field
from enum import Enum, auto
from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
from torch import nn, optim
import os

import torch


class ModelType(Enum):
    CNN = auto()
    MLP = auto()
    LOGISTIC_REGRESSION = auto()


class LearningType(Enum):
    SUPERVISED = auto()
    SEMI_SUPERVISED = auto()


class BoardRepresentation(Enum):
    PIECE_INDEX = (1, 8, 8)
    PIECE_TYPES = (6, 8, 8)
    PIECE_TYPES_AND_COLORS = (12, 8, 8)


class PuzzleRepresentation(Enum):
    FIRST = 1
    FIRST_AND_LAST = 2
    ALL = 4


@dataclass
class Config:
    # Directories and files paths
    ENGINES_DIRECTORY: str = "engines/"
    DATA_DIRECTORY: str = "data/"
    MODELS_DIRECTORY: str = "models/"
    LOGS_DIRECTORY: str = "logs/"
    PLOTS_DIRECTORY: str = "plots/"
    GAMES_DIRECTORY: str = field(default="data/games", init=False)
    PUZZLES_DIRECTORY: str = field(default="data/puzzles", init=False)
    OPENING_BOOK_PATH: str = field(default="data/codekiddy.bin", init=False)
    ENGINE_PATHS: dict[str, str] = field(default_factory=dict, init=False)

    # Game simulation parameters
    SIMULATE_GAMES_LOG_PATH: str = os.path.join(LOGS_DIRECTORY, "simulate_games.log")
    CURRENT_ENGINES: tuple[str, str] = ("stockfish", "stockfish")  # (white, black)
    GAME_COUNT: int = 65536
    USE_GAME_TIME: bool = False
    GAME_TIME_SECONDS: int = 2
    INCREMENT_SECONDS: int = 0
    MOVE_DEPTH: int = 5

    # Puzzle analysis parameters
    FIND_PUZZLES_LOG_PATH: str = os.path.join(LOGS_DIRECTORY, "find_puzzles.log")
    PUZZLE_ANALYSIS_ENGINE: str = "stockfish"
    ANALYSIS_DEPTH: int = 16
    EVALUATION_THRESHOLD: int = 350
    MIN_MATERIAL_GAIN: int = 210
    MIN_WHITE_BETTER_THAN_NEXT_MOVE: int = 250
    PUZZLE_PLY: int = 3

    # Puzzle classification parameters
    CLASSIFY_PUZZLES_LOG_PATH: str = os.path.join(LOGS_DIRECTORY, "classify_puzzles.log")

    EVALUATE_LOG_PATH = os.path.join(LOGS_DIRECTORY, "evaluate.log")

    # Piece values for material evaluation
    PAWN_VALUE: int = 100
    KNIGHT_VALUE: int = 300
    BISHOP_VALUE: int = 300
    ROOK_VALUE: int = 500
    QUEEN_VALUE: int = 900
    KING_VALUE: int = 1000
    PIECE_VALUES: dict[int, int] = field(default_factory=dict, init=False)

    PIECE_INDICES_MAP: dict[str, int] = field(default_factory=lambda: {
        'p': -1, 'P': 1, 'n': -2, 'N': 2, 'b': -3, 'B': 3,
        'r': -4, 'R': 4, 'q': -5, 'Q': 5, 'k': -6, 'K': 6,
    })

    # Concurrency settings
    CPU_COUNT: int = field(default_factory=os.cpu_count)

    # Model parameters
    MODEL_TYPE: ModelType = ModelType.MLP

    PUZZLE_CLASSES: list[str] = field(default_factory=lambda: [
        'alignment',  # pins, skewers, discovered
        'fork',       # one piece attacks two or more targets
        'promotion',  # pawn promotions
    ])

    DATA_SPLIT: dict[str, float] = field(default_factory=lambda: {
        'train': 0.7, 'validate': 0.2, 'test': 0.1
    })
    NUM_SAMPLES: int | None = None
    NUM_UNLABELED_SAMPLES: int | None = None

    LEARNING_TYPE: LearningType = LearningType.SUPERVISED
    UNSUPERVISED_LOSS_WEIGHT: float = 0.35
    LABEL_CONFIDENCE_THRESHOLD: float = 0.95

    BOARD_REPRESENTATION: BoardRepresentation = BoardRepresentation.PIECE_TYPES_AND_COLORS
    PUZZLE_REPRESENTATION: PuzzleRepresentation = PuzzleRepresentation.ALL

    CONVOLUTION_LAYERS: list[tuple[int, int, int, int]] = field(
        default_factory=lambda: [(32, 3, 1, 1), (64, 3, 1, 1)])
    FULLY_CONNECTED_LAYERS: list[int] = field(default_factory=lambda: [64, 64])

    # Training hyperparameters
    ACTIVE_LEARNING_LOG_PATH: str = os.path.join(LOGS_DIRECTORY, "active_learning.log")
    TUNE_LOG_PATH: str = os.path.join(LOGS_DIRECTORY, "tune.log")
    TRAIN_LOG_PATH: str = os.path.join(LOGS_DIRECTORY, "train.log")
    RANDOM_SEED: int = 7
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    MAX_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_DELTA = 1e-6
    BATCH_SIZE: int = 48
    LEARNING_RATE: float = 0.002
    DROPOUT: float = 0.65
    ACTIVATION_FUNCTION: nn.Module = nn.Sigmoid

    LOSS_FUNCTION: nn.Module = nn.CrossEntropyLoss
    OPTIMIZATION_FUNCTION: optim.Optimizer = optim.Adam
    SCHEDULER: optim.lr_scheduler.LRScheduler | None = None

    def __post_init__(self):
        self.GAMES_DIRECTORY = os.path.join(self.DATA_DIRECTORY, 'games')
        self.PUZZLES_DIRECTORY = os.path.join(self.DATA_DIRECTORY, 'puzzles')
        self.OPENING_BOOK_PATH = os.path.join(self.DATA_DIRECTORY, "codekiddy.bin")
        self.ENGINE_PATHS = {
            "stockfish": os.path.join(self.ENGINES_DIRECTORY, "stockfish"),
            "lc0": os.path.join(self.ENGINES_DIRECTORY, "lc0"),
        }

        self.PIECE_VALUES = {
            PAWN: self.PAWN_VALUE,
            KNIGHT: self.KNIGHT_VALUE,
            BISHOP: self.BISHOP_VALUE,
            ROOK: self.ROOK_VALUE,
            QUEEN: self.QUEEN_VALUE,
            KING: self.KING_VALUE
        }

    @staticmethod
    def conv2d_output_size(size, kernel_size, stride, padding):
        return (size + 2 * padding - kernel_size) // stride + 1
