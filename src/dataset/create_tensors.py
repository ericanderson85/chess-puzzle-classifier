import os
import shutil
import torch
import chess
import chess.pgn
import matplotlib.pyplot as plt
import numpy as np


class DistributionInfo:
    def __init__(self):
        self.results = {"0": 0, "1": 0, "2": 0}
        self.engines = dict()
        self.ply = dict()
        self.material = dict()


DATA_DIRECTORY = "data/"
DISTRIBUTION_PLOTS_DIRECTORY = os.path.join(DATA_DIRECTORY, "plots")

MIN_PLY = 30


def get_board_tensor(board: chess.Board):
    if board.turn == chess.BLACK:
        board = board.mirror()

    board_matrix = [[[0 for _ in range(6)] for _ in range(8)] for _ in range(8)]
    for file in range(8):
        for rank in range(8):
            square = chess.square(file, rank)
            piece_type = board.piece_type_at(square)
            if piece_type is None:
                continue
            color = board.color_at(square)
            board_matrix[rank][file][piece_type - 1] = 1 if color == chess.WHITE else -1

    board_tensor = torch.as_tensor(board_matrix, dtype=torch.float32)
    board_tensor = board_tensor.permute(2, 0, 1)
    return board_tensor


PIECE_VALUE_MAP = [1, 3, 3, 5, 9, 0]


def sum_material(board: chess.Board):
    material = 0
    for file in range(8):
        for rank in range(8):
            square = chess.square(file, rank)
            piece_type = board.piece_type_at(square)
            if piece_type is None:
                continue
            material += PIECE_VALUE_MAP[piece_type - 1]
    return material


def update_distributions(
    distribution_info: DistributionInfo, board: chess.Board, game: chess.pgn.Game
):
    headers = game.headers
    distribution_info.engines[headers["White"]] = (
        distribution_info.engines.get(headers["White"], 0) + 1
    )
    distribution_info.engines[headers["Black"]] = (
        distribution_info.engines.get(headers["Black"], 0) + 1
    )
    distribution_info.ply[board.ply()] = distribution_info.ply.get(board.ply(), 0) + 1
    distribution_info.results[headers["Label"]] += 1
    material = sum_material(board)
    distribution_info.material[material] = (
        distribution_info.material.get(material, 0) + 1
    )


def save_distribution_plots(distributions):

    for phase in ["train", "validate", "test"]:
        phase_plots_directory = os.path.join(DISTRIBUTION_PLOTS_DIRECTORY, phase)
        os.mkdir(phase_plots_directory)

        distribution_info = distributions[phase]

        plt.bar(distribution_info.engines.keys(), distribution_info.engines.values())
        plt.title("Engines Distribution")
        plt.xlabel("Engine Type")
        plt.ylabel("Count")
        plt.savefig(os.path.join(phase_plots_directory, "engines.png"))
        plt.clf()

        plt.bar(distribution_info.results.keys(), distribution_info.results.values())
        plt.title("Results Distribution")
        plt.xlabel("Result Type")
        plt.ylabel("Count")
        plt.savefig(os.path.join(phase_plots_directory, "results.png"))
        plt.clf()

        ply = distribution_info.ply
        ply_keys = sorted(ply.keys())
        ply_values = [ply[x] for x in ply_keys]
        plt.bar(ply_keys, ply_values)
        plt.title("Ply Distribution")
        plt.xlabel("Ply Value")
        plt.ylabel("Count")
        plt.savefig(os.path.join(phase_plots_directory, "ply.png"))
        plt.clf()

        material = distribution_info.material
        material_keys = sorted(material.keys())
        material_values = [material[x] for x in material_keys]
        plt.bar(material_keys, material_values)
        plt.title("Material Distribution")
        plt.xlabel("Material Type")
        plt.ylabel("Count")
        plt.savefig(os.path.join(phase_plots_directory, "material.png"))
        plt.clf()


RESULTS = ["0-1", "1/2-1/2", "1-0"]


def create_tensors():
    distributions = dict()

    for phase in ["train", "validate", "test"]:
        distributions[phase] = DistributionInfo()

        phase_directory = os.path.join(DATA_DIRECTORY, phase)
        position_directories = os.listdir(phase_directory)
        position_directories.sort(key=lambda dir: int(dir))

        for id in position_directories:
            pgn_path = os.path.join(phase_directory, id, f"{id}.pgn")

            with open(pgn_path, "r") as pgn_file:
                game = chess.pgn.read_game(pgn_file)

            total_moves = len(list(game.mainline_moves()))
            min_ply = min(MIN_PLY, total_moves - 4)
            max_ply = total_moves - 4
            rand = round(np.random.normal(loc=65, scale=10))
            num_moves_to_make = min(max(min_ply, rand), max_ply)

            board = game.board()
            for move_number, move in enumerate(game.mainline_moves()):
                if move_number >= num_moves_to_make:
                    break
                board.push(move)

            if board.turn == chess.BLACK:
                game.headers["Label"] = str(2 - RESULTS.index(game.headers["Result"]))
                board = board.mirror()
            else:
                game.headers["Label"] = str(RESULTS.index(game.headers["Result"]))

            with open(pgn_path, "w") as pgn_file:
                exporter = chess.pgn.FileExporter(pgn_file)
                game.accept(exporter)

            update_distributions(distributions[phase], board, game)
            torch.save(
                get_board_tensor(board), os.path.join(phase_directory, id, f"{id}.pt")
            )

    save_distribution_plots(distributions)


def main():
    if os.path.exists(DISTRIBUTION_PLOTS_DIRECTORY):
        shutil.rmtree(DISTRIBUTION_PLOTS_DIRECTORY)
    os.mkdir(DISTRIBUTION_PLOTS_DIRECTORY)
    create_tensors()


if __name__ == "__main__":
    main()
