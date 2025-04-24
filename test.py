import os

from src.util.chess_util import get_game, write_game_to_file


def count_moves(game):
    return sum(1 for _ in game.mainline_moves())


def main(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            game = get_game(file_path, None)

            move_count = count_moves(game)

            if move_count == 3:
                continue
            if move_count != 4:
                os.remove(file_path)
                continue

            node = game
            while node.variations:
                node = node.variations[0]

            node.parent.variations.remove(node)

            write_game_to_file(game, file_path, None)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    main("data/puzzles")
