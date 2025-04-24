import sys
import os
import subprocess
import asyncio
from logging import Logger
from chess import Board, WHITE, BLACK, Move, pgn
from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
from chess.engine import UciProtocol, InfoDict, popen_uci

from src.util.config import (
    FIND_PUZZLES_LOG_PATH, GAMES_DIRECTORY, PIECE_VALUES, PUZZLE_ANALYSIS_ENGINE, CPU_COUNT, EVALUATION_THRESHOLD,
    MIN_MATERIAL_GAIN, MIN_WHITE_BETTER_THAN_NEXT_MOVE, ENGINE_PATHS, PUZZLE_PLY, PUZZLES_DIRECTORY
)
from src.util.chess_util import (
    get_material, get_top_lines
)
from src.util.logger import get_logger

engine_pool: asyncio.Queue = None


async def init_engine_pool(pool_size: int, logger: Logger) -> None:
    global engine_pool
    try:
        engine_pool = asyncio.Queue(maxsize=pool_size)
        for i in range(pool_size):
            try:
                _, engine = await popen_uci(ENGINE_PATHS[PUZZLE_ANALYSIS_ENGINE], stderr=subprocess.DEVNULL)
                await engine_pool.put(engine)
                logger.debug(f"Initialized engine {i+1}/{pool_size}")
            except Exception as e:
                logger.error(f"Failed to initialize engine {i+1}: {str(e)}")
                raise
        logger.info(f"Initialized engine pool with {pool_size} engines")
    except Exception as e:
        logger.error(
            f"Failed to initialize engine pool: {str(e)}", exc_info=True)
        raise


async def close_engine_pool(logger: Logger) -> None:
    global engine_pool
    if engine_pool is None:
        logger.warning("Engine pool is None, nothing to close")
        return

    engines = []
    try:
        while not engine_pool.empty():
            engines.append(await engine_pool.get())

        logger.info(f"Closing {len(engines)} engines...")
        for i, engine in enumerate(engines):
            try:
                await engine.quit()
                logger.debug(f"Closed engine {i+1}/{len(engines)}")
            except Exception as e:
                logger.warning(f"Error closing engine {i+1}: {str(e)}")
    except Exception as e:
        logger.error(
            f"Error during engine pool shutdown: {str(e)}", exc_info=True)


def is_significant_move_diff(info: list[InfoDict]) -> bool:
    if len(info) < 2:
        return False

    best = info[0]["score"].white()
    second = info[1]["score"].white()

    if best.is_mate():
        return False

    if second.is_mate():
        return second.mate() < 0

    return second.score() < EVALUATION_THRESHOLD and best.score() - second.score() >= MIN_WHITE_BETTER_THAN_NEXT_MOVE


async def get_puzzle_solution(
    engine: UciProtocol,
    board: Board,
    starting_material: int,
    ply: int,
    moves: list[Move],
    logger: Logger
) -> list[Move] | None:

    if board.is_game_over():
        return None

    try:
        if board.turn == WHITE:
            info = await get_top_lines(engine, board, 2)
            if info is None or len(info) < 2:
                return None

            best_move = info[0]["pv"][0]
            best_move_score = info[0]["score"].white()

            if best_move_score.is_mate() or best_move_score.score() < EVALUATION_THRESHOLD:
                return None

            if ply == 0:
                if board.is_capture(best_move) and not board.is_en_passant(best_move):
                    piece_moved = board.piece_type_at(best_move.from_square)
                    piece_taken = board.piece_type_at(best_move.to_square)

                    is_winning_capture = PIECE_VALUES[piece_taken] > PIECE_VALUES[piece_moved]
                    is_undefended = len(board.attackers(BLACK, best_move.to_square)) == 0
                    if is_winning_capture or is_undefended:
                        return None

                if best_move.promotion is not None and best_move.promotion == QUEEN:
                    return None

            if not is_significant_move_diff(info):
                return None

            if ply == PUZZLE_PLY:
                won_material = get_material(board) - starting_material >= MIN_MATERIAL_GAIN
                return moves if won_material else None

        else:
            info = await get_top_lines(engine, board, 1)
            if info is None:
                return None

            best_move = info[0]["pv"][0]
            best_move_score = info[0]["score"].black()

            if best_move_score.is_mate():
                return None

        board.push(best_move)
        moves.append(best_move)
        return await get_puzzle_solution(engine, board, starting_material, ply + 1, moves, logger)

    except Exception as e:
        logger.warning(
            f"Error in analysis: {type(e).__name__}: {str(e) or 'No message'} (FEN: {board.fen()})")
        return None


async def find_puzzle(engine: UciProtocol, game: pgn.Game, logger: Logger) -> tuple[str, list[Move]] | None:
    board = game.board()

    try:
        for move_number, move in enumerate(game.mainline_moves()):
            was_capture = board.is_capture(move)
            gave_check = board.gives_check(move)

            board.push(move)

            if move_number < 8 or was_capture or gave_check:
                continue

            white_perspective = board.copy() if board.turn == WHITE else board.mirror()
            fen = white_perspective.fen()

            solution = await get_puzzle_solution(
                engine,
                white_perspective,
                get_material(white_perspective),
                0,
                [],
                logger
            )

            if solution is None:
                continue

            return fen, solution

    except Exception as e:
        logger.warning(f"Error game at FEN: {board.fen()}: {str(e)}")

    return None


async def process_file(file_path: str, logger: Logger) -> None:
    if engine_pool is None:
        logger.error("Engine pool is not initialized")
        return

    engine = None
    try:
        engine = await engine_pool.get()

        logger.info(f"Processing file: {file_path}")

        try:
            with open(file_path) as pgn_file:
                game = pgn.read_game(pgn_file)
                if game is None:
                    logger.warning(f"Invalid or empty PGN file: {file_path}")
                    return
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return
        except Exception as e:
            logger.error(f"Error reading PGN file {file_path}: {str(e)}")
            return

        try:
            puzzle = await find_puzzle(engine, game, logger)
        except Exception as e:
            logger.error(
                f"Error finding puzzles in {file_path}: {str(e)}", exc_info=True)
            return

        if puzzle is None:
            return

        game_id = os.path.basename(file_path).replace(".pgn", "")
        puzzle_pgn_path = os.path.join(
            PUZZLES_DIRECTORY, f"{game_id}.pgn"
        )

        with open(puzzle_pgn_path, "w") as puzzle_pgn_file:
            fen, solution = puzzle

            pgn_game = pgn.Game()

            for header in ["White", "Black", "Date", "Event"]:
                if header in game.headers:
                    pgn_game.headers[header] = game.headers[header]

            pgn_game.headers["FEN"] = fen
            pgn_game.headers["Event"] = "Puzzle"

            node = pgn_game
            for move in solution:
                node = node.add_variation(move)
            pgn_game.headers["Result"] = "*"

            exporter = pgn.StringExporter(
                headers=True, variations=False, comments=False)
            game_str = pgn_game.accept(exporter)
            puzzle_pgn_file.write(game_str + "\n\n")

            logger.info(f"Saved puzzle to {puzzle_pgn_path}")

    except Exception as e:
        logger.error(
            f"Unexpected error processing {file_path}: {str(e)}", exc_info=True)
    finally:
        if engine is not None:
            try:
                await engine_pool.put(engine)
            except Exception as e:
                logger.error(f"Error returning engine to pool: {str(e)}")


async def main(logger: Logger) -> None:
    await init_engine_pool(CPU_COUNT, logger)

    try:
        games_start = 0
        games = os.listdir(GAMES_DIRECTORY)
        game_ids = sorted([game_path.replace('.pgn', '')
                          for game_path in games], key=lambda g: int(g))

        if os.path.exists(PUZZLES_DIRECTORY):
            puzzles = os.listdir(PUZZLES_DIRECTORY)
            puzzle_game_ids = [
                int(game_path.replace('.pgn', ''))
                for game_path in puzzles
                if game_path.replace('.pgn', '').isnumeric()
            ]

            if len(puzzle_game_ids) != 0:
                games_start = max(puzzle_game_ids) + 1
                logger.info(
                    f"Puzzles already exist. Starting from game {games_start}")
        else:
            os.mkdir(PUZZLES_DIRECTORY)

        logger.info(f"Finding puzzles in {len(games) - games_start} games...")
        tasks = []
        for game_id in game_ids[games_start:]:
            game_path = os.path.join(GAMES_DIRECTORY, f"{game_id}.pgn")
            if not os.path.exists(game_path):
                logger.warning(f"File {game_path} does not exist")
                continue
            tasks.append(process_file(game_path, logger))

        await asyncio.gather(*tasks)

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")

    finally:
        await close_engine_pool(logger)

if __name__ == "__main__":
    logger = get_logger(__name__, FIND_PUZZLES_LOG_PATH)
    try:
        asyncio.run(main(logger))
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)
