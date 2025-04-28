from logging import Logger
import random
from chess import Board, popcount, WHITE, BLACK, polyglot, pgn
from chess.engine import UciProtocol, InfoDict, Limit

from src.util.config import Config


async def get_top_lines(config: Config, engine: UciProtocol, board: Board, multipv: int) -> list[InfoDict]:
    info = await engine.analyse(
        board,
        Limit(depth=config.ANALYSIS_DEPTH),
        multipv=multipv,
    )

    assert info is not None, "Engine analysis returned None"

    return info


def get_material(config: Config, board: Board) -> int:
    white = board.occupied_co[WHITE]
    black = board.occupied_co[BLACK]
    difference = (config.PAWN_VALUE * (popcount(white & board.pawns) -
                                       popcount(black & board.pawns))
                  + config.KNIGHT_VALUE * (popcount(white & board.knights) -
                                           popcount(black & board.knights))
                  + config.BISHOP_VALUE * (popcount(white & board.bishops) -
                                           popcount(black & board.bishops))
                  + config.ROOK_VALUE * (popcount(white & board.rooks) -
                                         popcount(black & board.rooks))
                  + config.QUEEN_VALUE * (popcount(white & board.queens) -
                                          popcount(black & board.queens)))

    return difference


async def get_book_move(config: Config, board: Board, reader: polyglot.MemoryMappedReader, logger: Logger) -> polyglot.Entry | None:
    try:
        entries = list(reader.find_all(board))
        if not entries:
            return None

        if random.random() < 0.75 and sum(entry.weight for entry in entries) > 0:
            total_weight = sum(entry.weight for entry in entries)
            target = random.randint(0, total_weight - 1)
            current_weight = 0
            for entry in entries:
                current_weight += entry.weight
                if current_weight > target:
                    return entry
            return entries[-1]
        else:
            return random.choice(entries)
    except Exception as e:
        logger.warning(f"Error selecting book move: {str(e)}")
        return None


def get_game(config: Config, puzzle_path: str, logger: Logger | None) -> pgn.Game | None:
    debug = logger.error if logger is not None else print

    try:
        with open(puzzle_path, 'r') as pgn_file:
            game = pgn.read_game(pgn_file)
            if game is None:
                debug(f"Invalid or empty PGN file: {puzzle_path}")
                return None
            return game
    except FileNotFoundError:
        debug(f"File not found: {puzzle_path}")
        return None
    except Exception as e:
        debug(f"Error reading PGN file {puzzle_path}: {str(e)}")
        return None


def write_game_to_file(config: Config, game: pgn.Game, puzzle_path: str, logger: Logger | None) -> None:
    debug = logger.error if logger is not None else print

    try:
        with open(puzzle_path, 'w') as file:
            file.write(str(game) + "\n")
    except Exception as e:
        debug(f"Error writing PGN file {puzzle_path}: {str(e)}")
        return None
