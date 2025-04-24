from typing import List
from src.util.config import PIECE_VALUES
from chess import Board, Move
import os
from logging import Logger
import chess
from chess import pgn, Board, Move, WHITE, BLACK, Square
from src.util.config import PIECE_VALUES, PUZZLES_DIRECTORY, CLASSIFY_PUZZLES_LOG_PATH, PUZZLE_CLASSES
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


def convert_bitmask_to_squares(bitmask: int) -> List[int]:

    return [
        square
        for square in chess.SQUARES
        if (bitmask >> square) & 1
    ]


def get_piece_positions(
    board: chess.Board,
    piece_type: chess.PieceType,
    color: bool
) -> List[int]:

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
) -> Tuple[bool, Tuple[int, int], List[chess.PieceType]]:

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
        attacker_types: List[chess.PieceType]
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
        attacker_value = PIECE_VALUES[attacker_piece.piece_type]
        if attacker_value < moved_value:
            return False
        if attacker_value > moved_value and not defender_squares:
            return False
    return True


def find_fork_targets(
        board: chess.Board,
        attack_square: int,
        mover_color: bool,
        moved_value: int
) -> List[int]:

    opponent_color = not mover_color
    attacked_squares = convert_bitmask_to_squares(board.attacks(attack_square))
    fork_targets: List[int] = []
    for target_square in attacked_squares:
        target_piece = board.piece_at(target_square)
        if (not target_piece
                or target_piece.color != opponent_color
                or target_piece.piece_type == chess.PAWN):
            continue
        target_value = PIECE_VALUES[target_piece.piece_type]
        target_defenders = board.attackers(opponent_color, target_square)
        if not target_defenders or target_value > moved_value:
            fork_targets.append(target_square)
    return fork_targets


def has_fork_attack_move(
    board: chess.Board,
    origin_square: int,
    target_squares: List[int]
) -> bool:

    for move in board.legal_moves:
        if (move.from_square == origin_square
                and move.to_square in target_squares):
            return True
    return False


def has_defensive_capture(
        board: chess.Board,
        target_square: int,
        moved_value: int
) -> bool:
    for move in board.legal_moves:
        if move.to_square != target_square or not board.is_capture(move):
            continue
        attacker_piece = board.piece_at(move.from_square)
        if PIECE_VALUES[attacker_piece.piece_type] > moved_value:
            return True
    return False


def detect_alignment(board: chess.Board, move: chess.Move) -> bool:
    return False


def detect_alignment(board: Board, moves: List[Move]) -> bool:
    board1 = board.copy()
    board1.push(moves[0])
    return False


def detect_fork(board: Board, moves: List[Move]) -> bool:
    board1 = board.copy()
    board1.push(moves[0])
    moved_square = moves[0].to_square
    moved_piece = board1.piece_at(moved_square)
    if moved_piece is None:
        return False
    mover_color = moved_piece.color
    moved_value = PIECE_VALUES[moved_piece.piece_type]

    if not can_withstand_capture(board1, moved_square, mover_color, moved_value):
        return False

    fork_targets = find_fork_targets(
        board1, moved_square, mover_color, moved_value
    )
    if len(fork_targets) < 2:
        return False

    if board.is_capture(moves[1]) and moves[1].to_square == moved_square:
        board2 = board1.copy()
        board2.push(moves[1])
        capturer = board2.piece_at(moved_square)
        if capturer and PIECE_VALUES[capturer.piece_type] > moved_value:
            if board.is_capture(moves[2]) and moves[2].to_square == moved_square:
                return True
            return False

    if board.is_capture(moves[2]) and moves[2].from_square == moved_square and moves[2].to_square in fork_targets:
        return True

    return False


def detect_promotion(board: Board, moves: List[Move]) -> bool:
    if moves[0].promotion is not None or moves[2].promotion is not None:
        return True

    board.push(moves[0])
    board.push(moves[1])

    if board.piece_type_at(moves[2].from_square) == chess.PAWN and not board.is_capture(moves[2]):
        raise Exception('yes')

    return False


def classify_puzzle(
    game: chess.pgn.Game,
    logger,
    model_prediction: float | None = None
) -> str | None:
    fen = game.headers.get("FEN")
    if fen is None:
        raise ValueError("No FEN header")

    base_board = Board(fen)

    moves: List[Move] = []
    node = game
    for ply in range(3):
        if not node.variations:
            raise ValueError("Puzzle has fewer than 3 moves")
        node = node.variations[0]
        moves.append(node.move)

    is_alignment = detect_alignment(base_board, moves)
    is_fork = detect_fork(base_board, moves)
    try:
        is_promotion = detect_promotion(base_board, moves)
    except:
        print(str(game) + '\n')
        is_promotion = True

    if not (is_alignment or is_fork or is_promotion):
        return None

    labels = []
    if is_alignment:
        labels.append("alignment")
    if is_fork:
        labels.append("fork")
    if is_promotion:
        labels.append("promotion")

    return " ".join(labels)


def main():
    logger = get_logger(__name__, CLASSIFY_PUZZLES_LOG_PATH)

    unlabeled_count = 0
    alignment_count = 0
    fork_count = 0
    promotion_count = 0

    puzzle_ids = sorted([
        int(puzzle_id[:-4])
        for puzzle_id in os.listdir(PUZZLES_DIRECTORY)
        if puzzle_id.endswith('.pgn') and puzzle_id[:-4].isdigit()
    ], reverse=True)

    for puzzle_id in puzzle_ids:
        try:
            puzzle_path = os.path.join(PUZZLES_DIRECTORY, f'{puzzle_id}.pgn')
            game = get_game(puzzle_path, logger)

            if game is None:
                continue
            if 'Label' in game.headers:
                del game.headers['Label']

            logger.info(f'Attempting to classify puzzle {puzzle_id}')
            label = classify_puzzle(game, logger)
            if label is None:
                unlabeled_count += 1
                continue

            if 'alignment' in label:
                alignment_count += 1
            if 'fork' in label:
                fork_count += 1
            if 'promotion' in label:
                promotion_count += 1

            logger.info(f'Classified puzzle {puzzle_id} as {label}')
            game.headers['Label'] = label

            write_game_to_file(game, puzzle_path, logger)

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
