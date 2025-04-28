from src.util.config import Config
from chess import Board, Move
import os
from logging import Logger
import chess
from chess import pgn, Board, Move, WHITE, BLACK, Square
from src.util.chess_util import get_game, write_game_to_file
from src.util.logger import get_logger

import chess


import chess
from typing import Iterator, List, Tuple


def get_file_and_rank(square: int) -> Tuple[int, int]:
    return chess.square_file(square), chess.square_rank(square)


def generate_ray_squares(
    start_square: int,
    direction: Tuple[int, int]
) -> Iterator[int]:

    file_index, rank_index = get_file_and_rank(start_square)
    delta_file, delta_rank = direction
    next_file = file_index + delta_file
    next_rank = rank_index + delta_rank
    while 0 <= next_file < 8 and 0 <= next_rank < 8:
        yield chess.square(next_file, next_rank)
        next_file += delta_file
        next_rank += delta_rank


def convert_bitmask_to_squares(bitmask: int) -> list[int]:

    return [
        square
        for square in chess.SQUARES
        if (bitmask >> square) & 1
    ]


def get_piece_positions(
    board: chess.Board,
    piece_type: chess.PieceType,
    color: bool
) -> list[int]:

    return [
        square
        for square in chess.SQUARES
        if (piece := board.piece_at(square))
        and piece.piece_type == piece_type
        and piece.color == color
    ]


def is_clear_path(board: chess.Board, start: int, end: int) -> bool:
    df = abs(chess.square_file(end) - chess.square_file(start))
    dr = abs(chess.square_rank(end) - chess.square_rank(start))
    if not (df == 0 or dr == 0 or df == dr):
        return False
    mask = chess.between(start, end)
    return (mask & board.occupied) == 0


def get_alignment_and_direction(
    queen_square: int,
    target_square: int
) -> Tuple[bool, Tuple[int, int], list[chess.PieceType]]:

    if queen_square == target_square:
        return False, (0, 0), []

    queen_file, queen_rank = get_file_and_rank(queen_square)
    target_file, target_rank = get_file_and_rank(target_square)
    file_difference = target_file - queen_file
    rank_difference = target_rank - queen_rank

    if file_difference == 0:
        direction = (0, 1 if rank_difference > 0 else -1)
        return True, direction, [chess.ROOK, chess.QUEEN]

    if rank_difference == 0:
        direction = (1 if file_difference > 0 else -1, 0)
        return True, direction, [chess.ROOK, chess.QUEEN]

    if abs(file_difference) == abs(rank_difference):
        direction = (
            1 if file_difference > 0 else -1,
            1 if rank_difference > 0 else -1
        )
        return True, direction, [chess.BISHOP, chess.QUEEN]

    return False, (0, 0), []


def is_queen_pin_in_direction(
        board: chess.Board,
        piece_square: int,
        direction: Tuple[int, int],
        friendly_color: bool,
        attacker_types: list[chess.PieceType]
) -> bool:

    for ray_square in generate_ray_squares(piece_square, direction):
        ray_piece = board.piece_at(ray_square)
        if ray_piece:
            if ray_piece.color != friendly_color and ray_piece.piece_type in attacker_types:
                return True
            break
    return False


def is_piece_pinned_to_queen(
    board: chess.Board,
    piece_square: int,
    color: bool
) -> bool:

    queen_squares = get_piece_positions(board, chess.QUEEN, color)
    for queen_square in queen_squares:
        aligned, direction, attacker_types = get_alignment_and_direction(
            queen_square, piece_square
        )
        if not aligned:
            continue
        if not is_clear_path(board, queen_square, piece_square):
            continue
        if is_queen_pin_in_direction(
                board, piece_square, direction, color, attacker_types):
            return True
    return False


def is_pinned(
    board: chess.Board,
    square: int,
    color: bool = None
) -> bool:

    piece = board.piece_at(square)
    if piece is None:
        return False
    effective_color = color if color is not None else piece.color
    if board.is_pinned(effective_color, square):
        return True
    return is_piece_pinned_to_queen(board, square, effective_color)


def can_withstand_capture(
    config: Config,
    board: chess.Board,
    square: int,
    mover_color: bool,
    moved_value: int
) -> bool:

    opponent_color = not mover_color
    attacker_squares = board.attackers(opponent_color, square)
    defender_squares = board.attackers(mover_color, square)
    for attacker_square in attacker_squares:
        attacker_piece = board.piece_at(attacker_square)
        attacker_value = config.PIECE_VALUES[attacker_piece.piece_type]
        if attacker_value < moved_value:
            return False
        if attacker_value > moved_value and not defender_squares:
            return False
    return True


def find_fork_targets(
        config: Config,
        board: chess.Board,
        attack_square: int,
        mover_color: bool,
        moved_value: int
) -> list[int]:

    opponent_color = not mover_color
    attacked_squares = convert_bitmask_to_squares(board.attacks(attack_square))
    fork_targets: list[int] = []
    for target_square in attacked_squares:
        target_piece = board.piece_at(target_square)
        if (not target_piece
                or target_piece.color != opponent_color
                or target_piece.piece_type == chess.PAWN):
            continue
        target_value = config.PIECE_VALUES[target_piece.piece_type]
        target_defenders = board.attackers(opponent_color, target_square)
        if not target_defenders or target_value > moved_value:
            fork_targets.append(target_square)
    return fork_targets


def has_fork_attack_move(
    board: chess.Board,
    origin_square: int,
    target_squares: list[int]
) -> bool:

    for move in board.legal_moves:
        if (move.from_square == origin_square
                and move.to_square in target_squares):
            return True
    return False


def has_defensive_capture(
        config: Config,
        board: chess.Board,
        target_square: int,
        moved_value: int
) -> bool:
    for move in board.legal_moves:
        if move.to_square != target_square or not board.is_capture(move):
            continue
        attacker_piece = board.piece_at(move.from_square)
        if config.PIECE_VALUES[attacker_piece.piece_type] > moved_value:
            return True
    return False


def detect_alignment(
    config: Config,
    board: Board,
    moves: list[Move]
) -> bool:
    board1 = board.copy()
    board1.push(moves[0])

    moved_piece = board1.piece_at(moves[0].to_square)
    if moved_piece is None:
        return False

    mover_color = moved_piece.color
    opponent_color = not mover_color
    moved_square = moves[0].to_square
    moved_value = config.PIECE_VALUES[moved_piece.piece_type]

    tactic_victims = set()

    for square in chess.SQUARES:
        piece = board1.piece_at(square)
        if piece and piece.color == opponent_color:
            if is_pinned(board1, square, opponent_color):
                tactic_victims.add(square)

    for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP]:
        attacker_squares = get_piece_positions(board1, piece_type, mover_color)
        for attacker_square in attacker_squares:
            directions = []
            if piece_type in [chess.QUEEN, chess.ROOK]:
                directions.extend([(0, 1), (1, 0), (0, -1), (-1, 0)])
            if piece_type in [chess.QUEEN, chess.BISHOP]:
                directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

            for direction in directions:
                ray = list(generate_ray_squares(attacker_square, direction))

                victims = []
                for ray_square in ray:
                    piece = board1.piece_at(ray_square)
                    if piece:
                        if piece.color == opponent_color:
                            victims.append((ray_square, config.PIECE_VALUES[piece.piece_type]))
                        break

                if len(victims) >= 2 and victims[1][1] > victims[0][1]:
                    tactic_victims.add(victims[1][0])

    attacks_before = set()
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == mover_color and square != moves[0].from_square:
            attacks_before.update(convert_bitmask_to_squares(board.attacks_mask(square)))

    attacks_after = set()
    for square in chess.SQUARES:
        piece = board1.piece_at(square)
        if piece and piece.color == mover_color and square != moved_square:
            attacks_after.update(convert_bitmask_to_squares(board1.attacks_mask(square)))

    discovered_attacks = attacks_after - attacks_before
    for attack_square in discovered_attacks:
        piece = board1.piece_at(attack_square)
        if piece and piece.color == opponent_color:
            tactic_victims.add(attack_square)

    if not tactic_victims:
        return False

    for move_index in [1, 2]:
        next_move = moves[move_index]
        if next_move.to_square in tactic_victims and board1.is_capture(next_move):
            return True

    if len(moves) > 1 and moves[1].to_square == moved_square and board1.is_capture(moves[1]):
        attacker_square = moves[1].from_square
        attacker_piece = board1.piece_at(attacker_square)

        if attacker_piece and config.PIECE_VALUES[attacker_piece.piece_type] < moved_value:
            board2 = board1.copy()
            board2.push(moves[1])

            if len(moves) > 2 and moves[2].to_square == moved_square and board2.is_capture(moves[2]):
                recaptor_square = moves[2].from_square
                recaptor_piece = board2.piece_at(recaptor_square)

                if recaptor_piece and config.PIECE_VALUES[recaptor_piece.piece_type] > config.PIECE_VALUES[attacker_piece.piece_type]:
                    return True

    return False


def detect_fork(
        config: Config,
        board: Board,
        moves: list[Move]
) -> bool:

    board1 = board.copy()
    board1.push(moves[0])
    moved_square = moves[0].to_square
    moved_piece = board1.piece_at(moved_square)
    if moved_piece is None:
        return False
    mover_color = moved_piece.color
    moved_value = config.PIECE_VALUES[moved_piece.piece_type]

    if not can_withstand_capture(config, board1, moved_square, mover_color, moved_value):
        return False

    fork_targets = find_fork_targets(
        config, board1, moved_square, mover_color, moved_value
    )
    if len(fork_targets) < 2:
        return False

    if board.is_capture(moves[1]) and moves[1].to_square == moved_square:
        board2 = board1.copy()
        board2.push(moves[1])
        capturer = board2.piece_at(moved_square)
        if capturer and config.PIECE_VALUES[capturer.piece_type] > moved_value:
            if board.is_capture(moves[2]) and moves[2].to_square == moved_square:
                return True
            return False

    if board.is_capture(moves[2]) and moves[2].from_square == moved_square and moves[2].to_square in fork_targets:
        return True

    return False


def detect_promotion(moves: list[Move]) -> bool:
    if moves[0].promotion is not None or moves[2].promotion is not None:
        return True

    return False


def classify_puzzle(
    config: Config,
    game: chess.pgn.Game
) -> str | None:
    fen = game.headers.get("FEN")
    if fen is None:
        raise ValueError("No FEN header")

    board = Board(fen)

    moves: list[Move] = []
    node = game
    for _ in range(3):
        if not node.variations:
            raise ValueError("Puzzle has fewer than 3 moves")
        node = node.variations[0]
        moves.append(node.move)

    if detect_promotion(moves):
        return "promotion"
    if detect_fork(config, board.copy(), moves):
        return "fork"
    if detect_alignment(config, board.copy(), moves):
        return "alignment"

    return None


def main():
    config = Config()
    logger = get_logger(__name__, config.CLASSIFY_PUZZLES_LOG_PATH)

    unlabeled_count = 0
    alignment_count = 0
    fork_count = 0
    promotion_count = 0

    puzzle_ids = sorted([
        int(puzzle_id[:-4])
        for puzzle_id in os.listdir(config.PUZZLES_DIRECTORY)
        if puzzle_id.endswith('.pgn') and puzzle_id[:-4].isdigit()
    ], reverse=True)

    for puzzle_id in puzzle_ids:
        try:
            puzzle_path = os.path.join(config.PUZZLES_DIRECTORY, f'{puzzle_id}.pgn')
            game = get_game(config, puzzle_path, logger)

            if game is None:
                continue
            if 'Label' in game.headers:
                del game.headers['Label']

            label = classify_puzzle(config, game)
            if label is None:
                unlabeled_count += 1
                continue

            if label == 'alignment':
                alignment_count += 1
            if label == 'fork':
                fork_count += 1
            if label == 'promotion':
                promotion_count += 1

            logger.info(f'Classified puzzle {puzzle_id} as {label}')
            game.headers['Label'] = label

            write_game_to_file(config, game, puzzle_path, logger)

        except ValueError as e:
            logger.error(f'Value error in puzzle {puzzle_id}: {e}')
        except Exception as e:
            logger.error(f'Error classifying puzzle {puzzle_id}: {e}')

    print(
        f'Alignment puzzles: {alignment_count}\n'
        f'Fork puzzles: {fork_count}\n'
        f'Promotion puzzles: {promotion_count}\n'
        '\n'
        f'Unlabeled puzzles: {unlabeled_count}\n'
    )


if __name__ == '__main__':
    main()
