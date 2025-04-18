import os
import shutil

os.mkdir(os.path.join('data', 'puzzles'))
os.mkdir(os.path.join('data', 'games'))

id = 0
for phase in ['train', 'validate', 'test']:
    phase_dir = os.path.join('data', phase)

    subdirectories = os.listdir(phase_dir)
    for subdirectory in subdirectories:
        game_path = os.path.join(
            'data', phase, subdirectory, f'{subdirectory}.pgn')
        puzzles_path = os.path.join(
            'data', phase, subdirectory, f'{subdirectory}_puzzles.pgn')

        if not os.path.exists(game_path):
            raise Exception(f'{game_path} does not exist')

        new_game_path = os.path.join('data', 'games', f'{id}.pgn')
        shutil.move(game_path, new_game_path)

        if os.path.exists(puzzles_path):
            new_puzzles_path = os.path.join('data', 'puzzles', f'{id}.pgn')
            shutil.move(puzzles_path, new_puzzles_path)

        id += 1

    os.rmdir(phase_dir)
