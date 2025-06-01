import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.dkm_hybridsn import OilSpillDKMNet
from utils.param_count_and_summary import print_summary
import time
import numpy as np

#load your dataset

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OilSpillDKMNet(output_units=2).to(device)

    # Print summary and parameter count
    print_summary(model, input_shape=(1, 30, 5, 5), device=device.type)

    # Prepare dummy train/test datasets
    train_dataset = Dataset(num_samples=1000)
    test_dataset = Dataset(num_samples=200)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # Inference Time Estimation
    model.eval()
    input_tensor = torch.randn(1, 1, 30, 5, 5).to(device)
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model(input_tensor)
        end = time.time()
    print("Inference time per image:", (end - start) / 100)

if __name__ == "__main__":
    main()
