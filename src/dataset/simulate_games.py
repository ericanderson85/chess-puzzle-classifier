import os
import sys
import subprocess
import shutil
import asyncio
from logging import Logger
import time
from chess import engine, pgn, polyglot, Board, WHITE

from src.util.chess_util import get_book_move
from src.util.config import (
    CURRENT_ENGINES, GAME_COUNTS, GAME_TIME_SECONDS, INCREMENT_SECONDS,
    ENGINE_PATHS, DATA_DIRECTORY, MOVE_DEPTH, OPENING_BOOK_PATH, CPU_COUNT, SIMULATE_GAMES_LOG_PATH, USE_GAME_TIME
)
from src.util.logger import get_logger

semaphore = asyncio.Semaphore(CPU_COUNT)


def with_semaphore(func):
    async def wrapper(*args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)

    return wrapper


def save_game(board: Board, engine_names: tuple[str, str], phase: str, game_id: int, logger: Logger) -> int:
    try:
        game_directory = os.path.join(DATA_DIRECTORY, phase, str(game_id))
        os.makedirs(game_directory, exist_ok=True)
        pgn_path = os.path.join(game_directory, f"{game_id}.pgn")

        game = pgn.Game.from_board(board)
        game.headers["White"] = engine_names[0]
        game.headers["Black"] = engine_names[1]
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        game.headers["Event"] = f"Engine match - {phase}"
        game.headers["Round"] = str(game_id)

        result = board.result(claim_draw=True)
        game.headers["Result"] = result

        with open(pgn_path, "w") as pgn_file:
            exporter = pgn.FileExporter(pgn_file)
            game.accept(exporter)

        return game_id
    except Exception as e:
        logger.error(f"Failed to save game {game_id}: {str(e)}")
        raise


async def initialize_engines(game_id: int, logger: Logger) -> tuple[engine.UciProtocol, engine.UciProtocol, tuple[str, str], bool]:
    try:
        _, engine1 = await engine.popen_uci(
            ENGINE_PATHS[CURRENT_ENGINES[0]], stderr=subprocess.PIPE)

        _, engine2 = await engine.popen_uci(
            ENGINE_PATHS[CURRENT_ENGINES[1]], stderr=subprocess.PIPE)

        switch_order = game_id % 2 == 1
        engines = (engine2, engine1) if switch_order else (engine1, engine2)
        engine_names = (
            (CURRENT_ENGINES[1], CURRENT_ENGINES[0])
            if switch_order
            else (CURRENT_ENGINES[0], CURRENT_ENGINES[1])
        )

        return engines[0], engines[1], engine_names, switch_order
    except Exception as e:
        logger.error(
            f"Failed to initialize engines for game {game_id}: {str(e)}")
        raise


async def close_engines(engines: list[engine.UciProtocol], logger: Logger) -> None:
    for i, eng in enumerate(engines):
        if eng is None:
            continue

        try:
            await eng.quit()
        except Exception as e:
            logger.warning(f"Error closing engine {i+1}: {str(e)}")


async def play_book_phase(board: Board, game_id: int, logger: Logger) -> None:
    book_available = False
    reader = None

    try:
        reader = polyglot.open_reader(OPENING_BOOK_PATH)
        book_available = True

        while book_available and not board.is_game_over(claim_draw=True):
            book_move_entry = await get_book_move(board, reader, logger)
            if not book_move_entry:
                break

            move = book_move_entry.move
            board.push(move)

            logger.debug(f"Game {game_id}: Book move {move.uci()} played")
    except Exception as e:
        logger.warning(
            f"Failed to play from book for game {game_id}: {str(e)}")
    finally:
        if reader is not None:
            reader.close()


async def make_engine_move(
    board: Board,
    engines: tuple[engine.UciProtocol, engine.UciProtocol],
    engine_names: tuple[str, str],
    clocks: list[float],
    game_id: int,
    move_count: int,
    switch_order: bool,
    logger: Logger
) -> tuple[Board, list[float], bool]:
    turn = board.turn
    engine_index = 0 if (turn == WHITE) == (not switch_order) else 1

    if USE_GAME_TIME:
        search_limit = engine.Limit(
            white_clock=clocks[0] if turn == WHITE else clocks[1],
            white_inc=INCREMENT_SECONDS,
            black_clock=clocks[1] if turn == WHITE else clocks[0],
            black_inc=INCREMENT_SECONDS,
        )
    else:
        search_limit = engine.Limit(
            depth=MOVE_DEPTH
        )

    try:
        result = await engines[engine_index].play(
            board=board,
            limit=search_limit,
            info=engine.INFO_ALL
        )

        if USE_GAME_TIME:
            clock_index = 0 if turn == WHITE else 1
            time_used = result.info.get("time", 0)
            clocks[clock_index] -= time_used
            clocks[clock_index] += INCREMENT_SECONDS
            clocks[clock_index] = max(clocks[clock_index], 0.1)

        board.push(result.move)
        return board, clocks, True
    except asyncio.TimeoutError:
        logger.warning(
            f"Game {game_id}: Engine {engine_names[engine_index]} "
            f"timeout on move {move_count}"
        )

        if turn == WHITE:
            board.headers["Result"] = "0-1"
        else:
            board.headers["Result"] = "1-0"
        return board, clocks, False


async def play_game(
    board: Board,
    engines: tuple[engine.UciProtocol, engine.UciProtocol],
    engine_names: tuple[str, str],
    game_id: int,
    switch_order: bool,
    logger: Logger
) -> Board:
    clocks = [GAME_TIME_SECONDS, GAME_TIME_SECONDS]
    move_count = 0

    await play_book_phase(board, game_id, logger)

    while not board.is_game_over(claim_draw=True):
        move_count += 1

        board, clocks, move_success = await make_engine_move(
            board, engines, engine_names, clocks, game_id, move_count, switch_order, logger
        )

        if not move_success:
            break

    return board


@with_semaphore
async def simulate(phase: str, game_id: int, logger: Logger) -> int:
    engine1 = None
    engine2 = None
    start_time = time.time()

    try:
        engine1, engine2, engine_names, switch_order = await initialize_engines(game_id, logger)

        board = Board()

        board = await play_game(
            board,
            (engine1, engine2),
            engine_names,
            game_id,
            switch_order,
            logger
        )

        game_id = save_game(board, engine_names, phase, game_id, logger)

        duration = time.time() - start_time
        logger.info(
            f"Game {game_id} completed in {duration:.1f}s: "
            f"{engine_names[0]} vs {engine_names[1]}, "
            f"Result: {board.result(claim_draw=True)}"
        )

        return game_id

    except Exception as e:
        logger.error(f"Error in game {game_id}: {str(e)}")
        return -game_id

    finally:
        await close_engines([engine1, engine2], logger)


async def main(logger: Logger) -> None:
    try:
        game_id = 0
        start_time = time.time()
        total_games = sum(GAME_COUNTS.values())

        logger.info(
            f"Starting simulation of {total_games} games between "
            f"{CURRENT_ENGINES[0]} and {CURRENT_ENGINES[1]} "
            f"({GAME_TIME_SECONDS}s + {INCREMENT_SECONDS}s increment)"
        )

        for phase in ["train", "validate", "test"]:
            if GAME_COUNTS[phase] == 0:
                continue

            phase_directory = os.path.join(DATA_DIRECTORY, phase)
            if os.path.exists(phase_directory):
                logger.info(f"Removing existing {phase} directory")
                shutil.rmtree(phase_directory)
            os.makedirs(phase_directory, exist_ok=True)

            logger.info(
                f"\nSimulating {GAME_COUNTS[phase]} {phase} games at "
                f"{GAME_TIME_SECONDS}s + {INCREMENT_SECONDS}s increment "
                f"between {CURRENT_ENGINES[0]} and {CURRENT_ENGINES[1]}"
            )

            tasks = []
            for _ in range(GAME_COUNTS[phase]):
                tasks.append(simulate(phase, game_id, logger))
                game_id += 1

            results = await asyncio.gather(*tasks, return_exceptions=True)

            failures = [result for result in results if isinstance(
                result, Exception) or (isinstance(result, int) and result < 0)]
            if failures:
                logger.warning(f"Phase {phase}: {len(failures)} games failed")
                for failure in failures:
                    print(failure)

            logger.info(
                f"Phase {phase} completed: {GAME_COUNTS[phase] - len(failures)} successful games")

        total_time = time.time() - start_time
        logger.info(f"Simulation completed in {total_time:.1f} seconds")

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise


if __name__ == "__main__":
    log = get_logger(__name__, SIMULATE_GAMES_LOG_PATH)
    try:
        asyncio.run(main(log))
    except KeyboardInterrupt:
        log.info("Simulation interrupted by user")
    except Exception as e:
        log.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)
