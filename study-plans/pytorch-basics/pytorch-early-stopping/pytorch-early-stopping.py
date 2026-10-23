import torch
import torch.nn as nn
import copy

def train_with_early_stopping(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, max_epochs: int, patience: int) -> dict:
    """
    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.
    """
    train_loss = []
    val_loss = []
    best_loss = float('inf')
    patience_counter = 0
    for e in range(max_epochs):
        model.train()
        num_batch = 0
        acc_loss = 0
        for X, y in train_loader:
            pred = model(X)
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            acc_loss += loss.item()
            num_batch += 1
            optimizer.step()
        train_loss.append(acc_loss/num_batch)
        model.eval()
        acc_loss_val = 0
        num_batch_val = 0
        with torch.no_grad():
            for X_val, y_val in val_loader: 
                pred_val = model(X_val)
                loss = criterion(pred_val, y_val)
                acc_loss_val += loss.item()
                num_batch_val += 1
        avg_val_loss = acc_loss_val/num_batch_val
        val_loss.append(avg_val_loss)
        
        if avg_val_loss >= best_loss:
            patience_counter+=1
        elif avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        if patience_counter >= patience:
            break
        
    return {"train_losses" : train_loss, "val_losses":val_loss, "stopped_epoch":e+1}