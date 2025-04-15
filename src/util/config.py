import os

# Directories and files paths
ENGINES_DIRECTORY = "engines/"
DATA_DIRECTORY = "data/"
OPENING_BOOK_PATH = os.path.join(
    DATA_DIRECTORY, "polyglot-collection/codekiddy.bin")
ENGINE_PATHS = {
    "stockfish": os.path.join(ENGINES_DIRECTORY, "stockfish"),
    "lc0": os.path.join(ENGINES_DIRECTORY, "lc0"),
}


# Game simulation parameters
SIMULATE_GAMES_LOG_PATH = "logs/simulate_games.log"
CURRENT_ENGINES = ("stockfish", "stockfish")  # (white, black)
GAME_COUNTS = {"train": 600, "validate": 200, "test": 200}
GAME_TIME_SECONDS = 2
INCREMENT_SECONDS = 0

# Puzzle analysis parameters
FIND_PUZZLES_LOG_PATH = "logs/find_puzzles.log"
PUZZLE_ANALYSIS_ENGINE = "stockfish"
ANALYSIS_DEPTH = 8
EVAL_THRESHOLD = 200
MIN_MATERIAL_GAIN = 120
MIN_WHITE_BETTER_THAN_NEXT_MOVE = 100
MIN_PUZZLE_LENGTH_PLY = 3
MAX_PUZZLE_LENGTH_PLY = 10

# Piece values for material evaluation
PAWN_VALUE = 100
KNIGHT_VALUE = 320
BISHOP_VALUE = 330
ROOK_VALUE = 500
QUEEN_VALUE = 900
KING_VALUE = 0

# Concurrency settings
CPU_COUNT = os.cpu_count()
