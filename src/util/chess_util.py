from logging import Logger
import random
from chess import Board, popcount, WHITE, BLACK, polyglot
from chess.engine import UciProtocol, InfoDict, Limit

from src.util.config import (ANALYSIS_DEPTH, PAWN_VALUE, KNIGHT_VALUE,
                             BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE)


async def get_top_lines(engine: UciProtocol, board: Board, multipv: int) -> list[InfoDict]:
    info = await engine.analyse(
        board,
        Limit(depth=ANALYSIS_DEPTH),
        multipv=multipv,
    )

    assert info is not None, "Engine analysis returned None"

    return info


def get_material(board: Board) -> int:
    white = board.occupied_co[WHITE]
    black = board.occupied_co[BLACK]
    difference = (PAWN_VALUE * (popcount(white & board.pawns) -
                                popcount(black & board.pawns))
                  + KNIGHT_VALUE * (popcount(white & board.knights) -
                                    popcount(black & board.knights))
                  + BISHOP_VALUE * (popcount(white & board.bishops) -
                                    popcount(black & board.bishops))
                  + ROOK_VALUE * (popcount(white & board.rooks) -
                                  popcount(black & board.rooks))
                  + QUEEN_VALUE * (popcount(white & board.queens) -
                                   popcount(black & board.queens)))

    return difference


async def get_book_move(board: Board, reader: polyglot.MemoryMappedReader, logger: Logger) -> polyglot.Entry | None:
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
