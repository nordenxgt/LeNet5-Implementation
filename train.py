import argparse

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import trange

from model import LeNet5, LeNet5Modern
from dataloader import mnist_dataloader
from utils import plot_loss_accuracy, calculate_accuracy

def main(epochs: int, modern: bool):
    if modern:
        model = LeNet5Modern(in_channels=1, feature_channels=6, num_classes=10)
        loss_fn = nn.CrossEntropyLoss()
    else:
        model = LeNet5(num_classes=10)
        loss_fn = nn.NLLLoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_dataloader, test_dataloader = mnist_dataloader()

    model.to(device)
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    for epoch in trange(epochs):
        train_loss, train_acc = 0, 0
        test_loss, test_acc = 0, 0

        model.train()
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            
            train_loss += loss.item()
            if modern == "LeNet5Modern":
                train_acc += calculate_accuracy(F.softmax(y_pred, dim=1), y)
            else:
                train_acc += calculate_accuracy(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, y)

                if modern == "LeNet5Modern":
                    test_acc += calculate_accuracy(F.softmax(y_pred, dim=1), y)
                else:
                    test_acc += calculate_accuracy(y_pred, y)
                
                test_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        test_loss /= len(test_dataloader)
        train_acc /= len(train_dataloader)
        test_acc /= len(test_dataloader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.2f} Train Accuracy: {train_acc*100:.2f} | Test Loss: {test_loss:.2f} Test Accuracy: {test_acc*100:.2f}")
    
    plot_loss_accuracy(
        train_losses=train_losses, 
        test_losses=test_losses,
        train_accuracies=train_accuracies,
        test_accuracies=test_accuracies,
        save=True,
        modern=modern
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Script")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--modern", action="store_true")
    args = parser.parse_args()
    main(args.epochs, args.modern)
    