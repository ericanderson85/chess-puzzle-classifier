import os
import sys
import subprocess
import asyncio
import random
from logging import Logger
import time
from chess import engine, pgn, polyglot, Board, WHITE
from src.util.config import Config
from src.util.logger import get_logger
from src.util.chess_util import get_book_move, write_game_to_file

config = Config()
semaphore = asyncio.Semaphore(config.CPU_COUNT)


def with_semaphore(func):
    async def wrapper(*args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)

    return wrapper


def save_game(
    config: Config,
    board: Board,
    engine_names: tuple[str, str],
    game_id: int,
    logger: Logger,
) -> int:
    try:
        pgn_path = os.path.join(config.GAMES_DIRECTORY, f"{game_id}.pgn")

        game = pgn.Game.from_board(board)
        game.headers["White"] = engine_names[0]
        game.headers["Black"] = engine_names[1]
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        game.headers["Event"] = f"Engine match"
        game.headers["Round"] = str(game_id)

        result = board.result(claim_draw=True)
        game.headers["Result"] = result

        write_game_to_file(config, game, pgn_path, logger)

        return game_id
    except Exception as e:
        logger.error(f"Failed to save game {game_id}: {str(e)}")
        raise


async def initialize_engines(
    config: Config,
    game_id: int,
    logger: Logger
) -> tuple[engine.UciProtocol, engine.UciProtocol, tuple[str, str], bool]:

    try:
        _, engine1 = await engine.popen_uci(
            config.ENGINE_PATHS[config.CURRENT_ENGINES[0]],
            stderr=subprocess.PIPE,
        )

        _, engine2 = await engine.popen_uci(
            config.ENGINE_PATHS[config.CURRENT_ENGINES[1]],
            stderr=subprocess.PIPE,
        )

        switch_order = game_id % 2 == 1
        engines = (engine2, engine1) if switch_order else (engine1, engine2)
        engine_names = (
            (config.CURRENT_ENGINES[1], config.CURRENT_ENGINES[0])
            if switch_order
            else (config.CURRENT_ENGINES[0], config.CURRENT_ENGINES[1])
        )

        return engines[0], engines[1], engine_names, switch_order
    except Exception as e:
        logger.error(
            f"Failed to initialize engines for game {game_id}: {str(e)}"
        )
        raise


async def close_engines(
    engines: list[engine.UciProtocol],
    logger: Logger
) -> None:

    for i, eng in enumerate(engines):
        if eng is None:
            continue

        try:
            await eng.quit()
        except Exception as e:
            logger.warning(f"Error closing engine {i+1}: {str(e)}")


async def play_book_phase(
    config: Config,
    board: Board,
    game_id: int,
    logger: Logger
) -> None:

    book_available = False
    reader = None

    try:
        reader = polyglot.open_reader(config.OPENING_BOOK_PATH)
        book_available = True

        while book_available and not board.is_game_over(claim_draw=True):
            book_move_entry = await get_book_move(config, board, reader, logger)
            if not book_move_entry:
                break

            move = book_move_entry.move
            board.push(move)

            logger.debug(f"Game {game_id}: Book move {move.uci()} played")
    except FileNotFoundError:
        logger.warning(
            f"Opening book not found at {config.OPENING_BOOK_PATH}. Skipping book phase for game {game_id}."
        )
    except Exception as e:
        logger.warning(
            f"Failed to play from book for game {game_id}: {str(e)}"
        )
    finally:
        if reader is not None:
            reader.close()


async def make_engine_move(
    config: Config,
    board: Board,
    engines: tuple[engine.UciProtocol, engine.UciProtocol],
    engine_names: tuple[str, str],
    clocks: list[float],
    game_id: int,
    move_count: int,
    switch_order: bool,
    logger: Logger,
) -> tuple[Board, list[float], bool]:

    turn = board.turn
    engine_index = 0 if (turn == WHITE) == (not switch_order) else 1

    if config.USE_GAME_TIME:
        search_limit = engine.Limit(
            white_clock=clocks[0] if turn == WHITE else clocks[1],
            white_inc=config.INCREMENT_SECONDS,
            black_clock=clocks[1] if turn == WHITE else clocks[0],
            black_inc=config.INCREMENT_SECONDS,
        )
    else:
        search_limit = engine.Limit(depth=config.MOVE_DEPTH)

    try:
        move = None
        if random.random() < 0.15:
            legal_moves = list(board.generate_legal_moves())
            if not legal_moves:
                logger.warning(
                    f"Game {game_id}: No legal moves found for random move, but game not over?")
                return board, clocks, False
            move = random.choice(legal_moves)
            logger.debug(f"Game {game_id}: Playing random move {move.uci()}")
        else:
            result = await engines[engine_index].play(
                board=board, limit=search_limit, info=engine.INFO_ALL
            )
            move = result.move

            if config.USE_GAME_TIME:
                clock_index = 0 if turn == WHITE else 1
                time_used = result.info.get("time", 0)
                clocks[clock_index] -= time_used
                clocks[clock_index] += config.INCREMENT_SECONDS
                clocks[clock_index] = max(clocks[clock_index], 0.25)

        if move is None:
            logger.error(
                f"Game {game_id}: Failed to determine a move for {engine_names[engine_index]}")
            if turn == WHITE:
                board.set_result("0-1")
            else:
                board.set_result("1-0")
            return board, clocks, False

        board.push(move)
        return board, clocks, True
    except (engine.EngineError, engine.EngineTerminatedError, BrokenPipeError) as e:
        logger.error(
            f"Game {game_id}: Engine {engine_names[engine_index]} error on move {move_count}: {e}"
        )
        if turn == WHITE:
            board.set_result("0-1")
        else:
            board.set_result("1-0")
        return board, clocks, False
    except asyncio.TimeoutError:
        logger.warning(
            f"Game {game_id}: Engine {engine_names[engine_index]} "
            f"timeout on move {move_count}"
        )

        if turn == WHITE:
            board.set_result("0-1")
        else:
            board.set_result("1-0")
        return board, clocks, False
    except Exception as e:  # Catch broader exceptions during move generation
        logger.error(
            f"Game {game_id}: Unexpected error during move generation by {engine_names[engine_index]} on move {move_count}: {e}",
            exc_info=True
        )
        if turn == WHITE:
            board.set_result("0-1")
        else:
            board.set_result("1-0")
        return board, clocks, False


async def play_game(
    config: Config,  # Pass config
    board: Board,
    engines: tuple[engine.UciProtocol, engine.UciProtocol],
    engine_names: tuple[str, str],
    game_id: int,
    switch_order: bool,
    logger: Logger,
) -> Board:
    clocks = [config.GAME_TIME_SECONDS, config.GAME_TIME_SECONDS]
    move_count = 0

    await play_book_phase(config, board, game_id, logger)

    while not board.is_game_over(claim_draw=True):
        move_count += 1

        board, clocks, move_success = await make_engine_move(
            config,
            board,
            engines,
            engine_names,
            clocks,
            game_id,
            move_count,
            switch_order,
            logger,
        )

        if not move_success:
            logger.warning(f"Game {game_id}: Ending game due to move failure on move {move_count}.")
            break

    if not board.result(claim_draw=True):
        logger.warning(f"Game {game_id}: Game loop finished but no result set. Setting draw.")
        board.set_result("1/2-1/2")

    return board


@with_semaphore
async def simulate(
    config: Config,
    game_id: int,
    logger: Logger
) -> int:

    engine1 = None
    engine2 = None
    start_time = time.time()

    try:
        engine1, engine2, engine_names, switch_order = await initialize_engines(
            config, game_id, logger
        )

        board = Board()

        board = await play_game(
            config,
            board,
            (engine1, engine2),
            engine_names,
            game_id,
            switch_order,
            logger,
        )

        saved_game_id = save_game(config, board, engine_names, game_id, logger)

        duration = time.time() - start_time
        result_str = board.result(claim_draw=True)
        logger.info(
            f"Game {saved_game_id} completed in {duration:.1f}s: "
            f"{engine_names[0]} vs {engine_names[1]}, "
            f"Result: {result_str}"
        )

        return saved_game_id

    except Exception as e:
        logger.error(f"Error processing game {game_id}: {str(e)}", exc_info=True)
        return -game_id

    finally:
        await close_engines([engine1, engine2], logger)


async def main(config: Config, logger: Logger) -> None:
    try:
        game_id = 0
        start_time = time.time()

        if config.USE_GAME_TIME:
            logger.info(
                f"Starting simulation of {config.GAME_COUNT} games between "
                f"{config.CURRENT_ENGINES[0]} and {config.CURRENT_ENGINES[1]} "
                f"({config.GAME_TIME_SECONDS}s + {config.INCREMENT_SECONDS}s increment)"
            )
        else:
            logger.info(
                f"Starting simulation of {config.GAME_COUNT} games between "
                f"{config.CURRENT_ENGINES[0]} and {config.CURRENT_ENGINES[1]} "
                f"({config.MOVE_DEPTH} depth)"
            )

        if os.path.exists(config.GAMES_DIRECTORY):
            logger.info(
                f"Game directory {config.GAMES_DIRECTORY} already exists. Appending new games")
            try:
                current_games = os.listdir(config.GAMES_DIRECTORY)
                valid_game_ids = []
                for fname in current_games:
                    if fname.endswith(".pgn"):
                        basename = fname[:-4]
                        if basename.isdigit():
                            valid_game_ids.append(int(basename))

                if valid_game_ids:
                    game_id = max(valid_game_ids) + 1
                    logger.info(f"New games start at id {game_id}")
                else:
                    logger.info("No valid previous game IDs found. Starting from 0.")
                    game_id = 0
            except Exception as e:
                logger.error(
                    f"Error reading existing games from {config.GAMES_DIRECTORY}: {e}. Starting from game ID 0.")
                game_id = 0
        else:
            try:
                os.makedirs(config.GAMES_DIRECTORY)
                logger.info(f"Created games directory: {config.GAMES_DIRECTORY}")
            except OSError as e:
                logger.error(f"Failed to create games directory {config.GAMES_DIRECTORY}: {e}")
                return

        tasks = []
        for i in range(config.GAME_COUNT):
            tasks.append(simulate(config, game_id + i, logger))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        failures = []
        for i, result in enumerate(results):
            current_game_id = game_id + i
            if isinstance(result, Exception):
                failures.append((current_game_id, result))
                logger.error(f"Game {current_game_id} failed with exception: {result}")
            elif isinstance(result, int) and result < 0:
                failures.append(
                    (current_game_id, f"Simulation function returned error code {result}"))
                logger.error(f"Game {current_game_id} failed with internal error code: {result}")
            elif isinstance(result, int) and result >= 0:
                success_count += 1
                if result != current_game_id:
                    logger.warning(
                        f"Game simulation for ID {current_game_id} returned unexpected success ID {result}")
            else:
                failures.append((current_game_id, f"Unexpected return type: {type(result)}"))
                logger.error(
                    f"Game {current_game_id} failed with unexpected return type: {type(result)} value: {result}")

        if failures:
            logger.warning(
                f"{len(failures)} out of {config.GAME_COUNT} games failed."
            )
            for gid, err in failures:
                logger.warning(f"  Game {gid}: {err}")
        else:
            logger.info(f"All {config.GAME_COUNT} games completed successfully.")

        total_time = time.time() - start_time
        logger.info(
            f"Simulation of {success_count}/{config.GAME_COUNT} games completed in {total_time:.1f} seconds"
        )

    except Exception as e:
        logger.error(f"Fatal error in main simulation loop: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    log = get_logger(__name__, config.SIMULATE_GAMES_LOG_PATH)
    try:
        asyncio.run(main(config, log))
    except KeyboardInterrupt:
        log.info("Simulation interrupted by user")
    except Exception as e:
        log.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)
