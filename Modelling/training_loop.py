import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Candidatecreator
from model import model as lol
from tqdm import tqdm

def train():
    device = torch.device("cuda" if torch.cuda.is_available() 
                       else "mps" if torch.backends.mps.is_available() 
                       else "cpu")
    epochs = 50
    batch_size = 256
    lr = 1e-4

    train_set = Candidatecreator(
        jsonl_path="../Data/processed/train.jsonl",
        data_root="../Data/processed"
    )
    val_set = Candidatecreator(
        jsonl_path="../Data/processed/val.jsonl",
        data_root="../Data/processed"
    )

    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    net = lol().to(device)
    optimizer = optim.Adam(net.parameters(),lr=lr)
    criterion = nn.BCELoss()
    best = float('inf')
    for epoch in range(epochs):
        net.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs,patches,labels = batch
            patches = patches.to(device)
            imgs = imgs.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = net(patches,imgs).squeeze(1)
            loss = criterion(outputs,labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
        net.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in val_loader:
                patches,imgs,labels = batch
                patches = patches.to(device)
                imgs = imgs.to(device)
                labels = labels.to(device).float()
                outputs = net(patches,imgs).squeeze(1)
                loss = criterion(outputs,labels)
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Val Loss: {total_loss/len(val_loader)}")
    val_loss = total_loss/len(val_loader)
    if val_loss < best:
        best = val_loss
        torch.save(net.state_dict(),'best_model.pth')
    print("Model saved")

if __name__ == '__main__':
    train()