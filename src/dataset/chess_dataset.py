import os
import chess.pgn
import torch
from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, phase_directory):    
        self.labels = []
        self.boards = []
        
        ids = os.listdir(phase_directory)
        ids.sort(key=lambda position_id: int(position_id))
    
        for id in ids:
            position_directory = os.path.join(phase_directory, id)
            
            pgn_path = os.path.join(position_directory, f'{id}.pgn')
            tensor_path = os.path.join(position_directory, f'{id}.pt')
            
            with open(pgn_path, 'r') as pgn_file:
                game = chess.pgn.read_game(pgn_file)
            self.labels.append(int(game.headers['Label']))
            self.boards.append(torch.load(tensor_path, weights_only=False))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.boards[idx], self.labels[idx]