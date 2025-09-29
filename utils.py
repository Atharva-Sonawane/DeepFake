# utils.py

import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime

def compute_metrics(preds, targets):
    preds = torch.sigmoid(preds).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    preds = (preds > 0.5).astype(int)

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    return acc, prec, rec, f1


class AverageMeter:
    """Tracks and averages loss/metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, epoch, path, best_f1=0):
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'best_f1': best_f1
    }, path)

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint


def log_epoch(epoch, loss, acc, prec, rec, f1, mode="Train"):
    print(f"[{mode}][Epoch {epoch:03d}] Loss: {loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
