import os
from logging import Logger
from chess import pgn
import pyperclip

from src.util.config import PUZZLES_DIRECTORY, CLASSIFY_PUZZLES_LOG_PATH
from src.util.chess_util import get_game, write_game_to_file
from src.util.logger import get_logger


PUZZLE_CLASSES = [
    'alignment',  # pins, skewers, xray, discovered
    'exchange',   # trading pieces correctly
    'fork',       # one piece attacks two or more targets
    'pawn',       # promotons, underpromotions, pawn pushes
    'sacrifice',  # loss of material for a greater gain
    'trapped',    # captured piece has nowhere to go
]


def classify_puzzle(game: pgn.Game, logger: Logger) -> str | None:
    pgn_string = str(game)
    try:
        pyperclip.copy(pgn_string)
        print("PGN copied to clipboard")
    except Exception as e:
        print(f"Copy failed: {e}")
        return None

    print("\nClassify as:")
    for puzzle_class in PUZZLE_CLASSES:
        print(f"  {puzzle_class[0]} - {puzzle_class}")
    print("\tEnter - Skip puzzle")

    user_input = input("\nEnter choice: ").strip().lower()

    if not user_input:
        logger.info("Puzzle skipped")
        return None

    for puzzle_class in PUZZLE_CLASSES:
        if puzzle_class.startswith(user_input):
            return puzzle_class

    logger.warning(f"Invalid input: {user_input}")
    return None


def main():
    logger = get_logger(__name__, CLASSIFY_PUZZLES_LOG_PATH)

    for puzzle_path in os.listdir(PUZZLES_DIRECTORY):
        try:
            file_name, file_extension = os.path.splitext(puzzle_path)
            if file_extension != '.pgn':
                continue

            game = get_game(os.path.join(
                PUZZLES_DIRECTORY, puzzle_path), logger)
            if game is None:
                continue

            logger.info(f'Attempting to classify puzzle {file_name}')
            label = classify_puzzle(game, logger)
            if label is None:
                continue

            logger.info(f'Classified puzzle {file_name} as {label}')
            game.headers['Label'] = label

            write_game_to_file(game, os.path.join(
                PUZZLES_DIRECTORY, puzzle_path), logger)

        except Exception as e:
            logger.error(f'Error classifying puzzle {puzzle_path}: {e}')


if __name__ == '__main__':
    main()
