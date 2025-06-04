import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.dataset_comma2k19 import Comma2k19Dataset
from models.model_rgb_bc import RGBBCNet
from torchvision import transforms

def train(model, train_loader, val_loader, device, epochs=20, lr=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            output = torch.stack(pred, dim=1)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                pred = model(imgs)
                output = torch.stack(pred, dim=1)
                val_loss += criterion(output, targets).item()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("Best model saved.")

if __name__ == '__main__':
    train_dataset = Comma2k19Dataset('metadata_train.csv', transform=transforms.ToTensor())
    val_dataset = Comma2k19Dataset('metadata_val.csv', transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGBBCNet().to(device)

    train(model, train_loader, val_loader, device)
