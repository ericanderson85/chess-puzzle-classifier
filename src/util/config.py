from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
import os

# Directories and files paths
ENGINES_DIRECTORY = "engines/"
DATA_DIRECTORY = "data/"
GAMES_DIRECTORY = os.path.join(DATA_DIRECTORY, 'games')
PUZZLES_DIRECTORY = os.path.join(DATA_DIRECTORY, 'puzzles')
OPENING_BOOK_PATH = os.path.join(DATA_DIRECTORY, "codekiddy.bin")
ENGINE_PATHS = {
    "stockfish": os.path.join(ENGINES_DIRECTORY, "stockfish"),
    "lc0": os.path.join(ENGINES_DIRECTORY, "lc0"),
}


# Game simulation parameters
SIMULATE_GAMES_LOG_PATH = "logs/simulate_games.log"
CURRENT_ENGINES = ("stockfish", "stockfish")  # (white, black)
GAME_COUNT = 65536
USE_GAME_TIME = False
GAME_TIME_SECONDS = 2
INCREMENT_SECONDS = 0
MOVE_DEPTH = 5

# Puzzle analysis parameters
FIND_PUZZLES_LOG_PATH = "logs/find_puzzles.log"
PUZZLE_ANALYSIS_ENGINE = "stockfish"
ANALYSIS_DEPTH = 16
EVAL_THRESHOLD = 300
MIN_MATERIAL_GAIN = 210
MIN_WHITE_BETTER_THAN_NEXT_MOVE = 250
MIN_PUZZLE_LENGTH_PLY = 3
MAX_PUZZLE_LENGTH_PLY = 32

# Piece values for material evaluation
PAWN_VALUE = 100
KNIGHT_VALUE = 300
BISHOP_VALUE = 300
ROOK_VALUE = 500
QUEEN_VALUE = 900
KING_VALUE = 1000
PIECE_VALUES = {
    PAWN: PAWN_VALUE,
    KNIGHT: KNIGHT_VALUE,
    BISHOP: BISHOP_VALUE,
    ROOK: ROOK_VALUE,
    QUEEN: QUEEN_VALUE,
    KING: KING_VALUE
}

# Concurrency settings
CPU_COUNT = os.cpu_count()
