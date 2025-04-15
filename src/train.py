import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.chess_cnn import ChessCNN
from dataset.chess_dataset import ChessDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_PHASE_DIRECTORY = 'data/train'
VALIDATE_PHASE_DIRECTORY = 'data/validate'
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_cost = 0.0
    correct = 0

    for boards, labels in train_loader:
        boards, labels = boards.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(boards)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
    
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_cost += loss.item() * boards.size(0)
    
    total_samples = len(train_loader.dataset)
    average_cost = total_cost / total_samples
    accuracy = correct / total_samples
    return average_cost, accuracy

def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_cost = 0.0
    correct = 0

    with torch.no_grad():
        for boards, labels in val_loader:
            boards, labels = boards.to(device), labels.to(device)
            outputs = model(boards)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            loss = loss_fn(outputs, labels)
            total_cost += loss.item() * boards.size(0)
    
    total_samples = len(val_loader.dataset)
    average_cost = total_cost / total_samples
    accuracy = correct / total_samples
    return average_cost, accuracy

def gradient_descent(model, train_loader, val_loader, optimizer, loss_fn, device):
    val_loss, val_accuracy = validate(model, val_loader, loss_fn, DEVICE)
    print(
        f'Epoch [0/{NUM_EPOCHS}] | '
        f'Validation Cost: {val_loss:.4f} | '
        f'Validation Accuracy: {(val_accuracy * 100):.2f}%'
    )
    for epoch in range(NUM_EPOCHS):
        train_cost, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_cost, val_accuracy = validate(model, val_loader, loss_fn, DEVICE)

        print(
            f'Epoch [{epoch+1}/{NUM_EPOCHS}] | '
            f'Training Cost: {train_cost:.4f} | '
            f'Training Accuracy: {(train_accuracy * 100):.2f}% | '
            f'Validation Cost: {val_cost:.4f} | '
            f'Validation Accuracy: {(val_accuracy * 100):.2f}%'
        )
    
def main():
    train_dataset = ChessDataset(phase_directory=TRAIN_PHASE_DIRECTORY)
    val_dataset = ChessDataset(phase_directory=VALIDATE_PHASE_DIRECTORY)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ChessCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    gradient_descent(model, train_loader, val_loader, optimizer, loss_fn, DEVICE)



if __name__ == '__main__':
    main()
