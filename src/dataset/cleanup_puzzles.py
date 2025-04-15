import os

from util.config import DATA_DIRECTORY


def main():
    for phase in ["train", "validate", "test"]:
        dir = os.path.join("data", phase)
        for game_id in os.listdir(dir):
            puzzles_path = os.path.join(
                DATA_DIRECTORY, game_id, f"{game_id}_puzzles.pgn")
            if os.path.exists(puzzles_path):
                os.remove(puzzles_path)


if __name__ == '__main__':
    main()
