import sys
import os
import subprocess
import asyncio
from logging import Logger
from chess import Board, WHITE, BLACK, Move, pgn
from chess.engine import UciProtocol, InfoDict, popen_uci

from src.util.config import (
    FIND_PUZZLES_LOG_PATH, PUZZLE_ANALYSIS_ENGINE, CPU_COUNT, DATA_DIRECTORY, ANALYSIS_DEPTH, EVAL_THRESHOLD,
    MIN_MATERIAL_GAIN, MIN_WHITE_BETTER_THAN_NEXT_MOVE, MIN_PUZZLE_LENGTH_PLY,
    MAX_PUZZLE_LENGTH_PLY, ENGINE_PATHS
)
from src.util.chess_util import (
    get_material, get_top_lines
)
from src.util.logger import get_logger

engine_pool: asyncio.Queue = None


def should_stop_searching(board: Board, ply: int) -> bool:
    return board.is_game_over() or ply > MAX_PUZZLE_LENGTH_PLY


def is_significant_move_diff(board: Board, info: list[InfoDict]) -> bool:
    if board.turn == BLACK:
        return True

    if len(info) < 2:
        return False

    best = info[0]
    second = info[1]
    if best["score"].is_mate():
        first_mate = best["score"].white().mate()
        if first_mate < 0:
            return False

        second_mate = best["score"].white().mate()
        if second_mate is not None and second_mate > 0:
            return False

        return True

    if second["score"].is_mate():
        return True

    try:
        best_score = best["score"].white().score()
        second_score = second["score"].white().score()
        return best_score - second_score >= MIN_WHITE_BETTER_THAN_NEXT_MOVE
    except Exception:
        return False


def is_valid_puzzle(board: Board, starting_material: int, ply: int) -> bool:
    if board.turn == BLACK:
        return False
    if ply < MIN_PUZZLE_LENGTH_PLY:
        return False

    if get_material(board) - starting_material < MIN_MATERIAL_GAIN:
        return False

    return True


async def get_puzzle_solution(
    engine: UciProtocol,
    board: Board,
    starting_material: int,
    ply: int,
    moves: list[Move],
    logger: Logger
) -> list[Move] | None:
    if should_stop_searching(board, ply):
        return None

    num_lines = 2 if board.turn == WHITE else 1

    try:
        info = await get_top_lines(engine, board, num_lines)

        if len(info) < num_lines:
            return None

        if not is_significant_move_diff(board, info):
            return None

        best_move = info[0]["pv"][0]
        moves.append(best_move)
        board.push(best_move)

        if is_valid_puzzle(board, starting_material, ply):
            return moves

        return await get_puzzle_solution(
            engine, board, starting_material, ply + 1, moves, logger
        )

    except Exception as e:
        logger.warning(f"Error in analysis: {str(e)}")
        return None


async def find_puzzles(engine: UciProtocol, game: pgn.Game, logger: Logger) -> list[tuple[str, list[Move]]]:
    board = game.board()
    puzzles = []

    skip_next_move = False

    for move_number, move in enumerate(game.mainline_moves()):
        try:
            skip = skip_next_move or move_number < 8
            skip_next_move = board.is_capture(move) or board.gives_check(move)
            board.push(move)

            if skip:
                continue

            white_perspective = (
                board.copy() if board.turn == WHITE else board.mirror()
            )

            fen = white_perspective.fen()

            info = await get_top_lines(engine, white_perspective, multipv=1)
            if not info:
                logger.debug(f"No analysis at move {move_number+1}")
                continue

            best_move = info[0]
            if best_move["score"].is_mate():
                logger.debug(
                    f"Skipping mate position at move {move_number+1}")
                continue

            if best_move["score"].white().score() < EVAL_THRESHOLD:
                logger.debug(
                    f"Position at move {move_number+1} below threshold")
                continue

            logger.debug(f"Looking for puzzle at move {move_number+1}")
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

            puzzles.append((fen, solution))
        except Exception as e:
            logger.warning(f"Error processing move {move_number+1}: {str(e)}")
            continue

    return puzzles


async def init_engine_pool(pool_size: int, logger: Logger) -> None:
    global engine_pool
    try:
        engine_pool = asyncio.Queue(maxsize=pool_size)
        for i in range(pool_size):
            try:
                _, engine = await popen_uci(
                    ENGINE_PATHS[PUZZLE_ANALYSIS_ENGINE], stderr=subprocess.DEVNULL
                )
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
            puzzles = await find_puzzles(engine, game, logger)
        except Exception as e:
            logger.error(
                f"Error finding puzzles in {file_path}: {str(e)}", exc_info=True)
            return

        try:
            game_id = os.path.basename(file_path).replace(".pgn", "")
            puzzle_pgn_path = os.path.join(
                os.path.dirname(file_path), f"{game_id}_puzzles.pgn"
            )

            if puzzles:
                with open(puzzle_pgn_path, "w") as puzzle_pgn_file:
                    for fen, solution in puzzles:
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
                            headers=True, variations=False, comments=False
                        )
                        game_str = pgn_game.accept(exporter)
                        puzzle_pgn_file.write(game_str + "\n\n")

                logger.info(
                    f"Saved {len(puzzles)} puzzles to {puzzle_pgn_path}")
        except Exception as e:
            logger.error(f"Error saving puzzles from {file_path}: {str(e)}")

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
        tasks = []

        for phase in ["train", "validate", "test"]:
            logger.info(f"Finding {phase} puzzles ...")
            for game_id in sorted(
                os.listdir(os.path.join(DATA_DIRECTORY, phase)), key=lambda g: int(g)
            ):
                game_path = os.path.join(
                    DATA_DIRECTORY, phase, game_id, f"{game_id}.pgn")
                if not os.path.exists(game_path):
                    logger.warning(f"File {game_path} does not exist")
                    continue
                tasks.append(process_file(game_path, logger))

        await asyncio.gather(*tasks)

    except Exception as e:
        logger.error("An error occurred during processing:", e)

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
