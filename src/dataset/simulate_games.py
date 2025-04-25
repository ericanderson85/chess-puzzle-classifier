import os
import sys
import subprocess
import asyncio
import random
from logging import Logger
import time
from chess import engine, pgn, polyglot, Board, WHITE

from src.util.chess_util import get_book_move
from src.util.config import (
    CURRENT_ENGINES, GAME_COUNT, GAME_TIME_SECONDS, GAMES_DIRECTORY, INCREMENT_SECONDS,
    ENGINE_PATHS, MOVE_DEPTH, OPENING_BOOK_PATH, CPU_COUNT, SIMULATE_GAMES_LOG_PATH, USE_GAME_TIME
)
from src.util.logger import get_logger

semaphore = asyncio.Semaphore(CPU_COUNT)


def with_semaphore(func):
    async def wrapper(*args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)

    return wrapper


def save_game(board: Board, engine_names: tuple[str, str], game_id: int, logger: Logger) -> int:
    try:
        pgn_path = os.path.join(GAMES_DIRECTORY, f"{game_id}.pgn")

        game = pgn.Game.from_board(board)
        game.headers["White"] = engine_names[0]
        game.headers["Black"] = engine_names[1]
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        game.headers["Event"] = f"Engine match"
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
        move = None
        if random.random() < 0.15:
            move = random.choice(list(board.generate_legal_moves()))
        else:
            result = await engines[engine_index].play(
                board=board,
                limit=search_limit,
                info=engine.INFO_ALL
            )
            move = result.move

            if USE_GAME_TIME:
                clock_index = 0 if turn == WHITE else 1
                time_used = result.info.get("time", 0)
                clocks[clock_index] -= time_used
                clocks[clock_index] += INCREMENT_SECONDS
                clocks[clock_index] = max(clocks[clock_index], 0.1)

        board.push(move)
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
async def simulate(game_id: int, logger: Logger) -> int:
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

        game_id = save_game(board, engine_names, game_id, logger)

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

        if USE_GAME_TIME:
            logger.info(
                f"Starting simulation of {GAME_COUNT} games between "
                f"{CURRENT_ENGINES[0]} and {CURRENT_ENGINES[1]} "
                f"({GAME_TIME_SECONDS}s + {INCREMENT_SECONDS}s increment)"
            )
        else:
            logger.info(
                f"Starting simulation of {GAME_COUNT} games between "
                f"{CURRENT_ENGINES[0]} and {CURRENT_ENGINES[1]} "
                f"({MOVE_DEPTH} depth)"
            )

        if os.path.exists(GAMES_DIRECTORY):
            logger.info(
                f"Game directory already exists. Appending new games")
            current_games = os.listdir(GAMES_DIRECTORY)
            current_games = list(
                map(lambda game_str: game_str.replace('.pgn', ''), current_games))
            current_games = list(
                filter(lambda game: game.isnumeric(), current_games))
            current_games = list(
                map(lambda game_str: int(game_str), current_games))
            if current_games:
                game_id = max(current_games) + 1
                logger.info(f"New games start at id {game_id}")
        else:
            os.makedirs(GAMES_DIRECTORY)

        tasks = []
        for _ in range(GAME_COUNT):
            tasks.append(simulate(game_id, logger))
            game_id += 1

        results = await asyncio.gather(*tasks, return_exceptions=True)

        failures = [result for result in results if isinstance(
            result, Exception) or (isinstance(result, int) and result < 0)]
        if failures:
            logger.warning(f"{len(failures)} games failed")
            for failure in failures:
                print(failure)

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
