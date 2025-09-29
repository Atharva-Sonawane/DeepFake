#train_model.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F

import config
from model import HybridEfficientNet
from data import create_dataloaders
from utils import compute_metrics, AverageMeter, save_checkpoint, log_epoch, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return loss.mean()

def train():
    # Config
    checkpoint_path = "Efficientb5_model_1.pth"
    num_epochs = 20

    # Data
    train_loader, val_loader = create_dataloaders(
        train_dir=config.TRAIN_DIR,
        val_dir=config.VAL_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_size=config.IMG_SIZE
    )

    # Model
    model = HybridEfficientNet(num_classes=1, use_temporal=False).to(device)

    # Loss
    criterion = nn.BCEWithLogitsLoss()
    focal = FocalLoss()

    # Optimizer
    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    scaler = GradScaler()

    # Resume from checkpoint if exists
    start_epoch = 1
    best_f1 = 0
    if os.path.exists(checkpoint_path):
        print("🔄 Resuming from checkpoint...")
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['best_f1']
        print(f"✅ Resumed from epoch {start_epoch} with best F1 = {best_f1:.4f}")

    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss_meter = AverageMeter()
        all_preds, all_targets = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) + 0.5 * focal(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_meter.update(loss.item(), images.size(0))
            all_preds.append(outputs)
            all_targets.append(labels)

            acc, _, _, _ = compute_metrics(outputs, labels)
            pbar.set_postfix(Loss=train_loss_meter.avg, Acc=acc * 100)

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        acc, prec, rec, f1 = compute_metrics(preds, targets)
        log_epoch(epoch, train_loss_meter.avg, acc, prec, rec, f1, mode="Train")

        # Validation
        model.eval()
        val_loss_meter = AverageMeter()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels) + 0.5 * focal(outputs, labels)

                val_loss_meter.update(loss.item(), images.size(0))
                all_preds.append(outputs)
                all_targets.append(labels)

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        acc, prec, rec, f1 = compute_metrics(preds, targets)
        log_epoch(epoch, val_loss_meter.avg, acc, prec, rec, f1, mode="Val")

        # Save checkpoint if improved
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(model, optimizer, epoch, checkpoint_path, best_f1)
            print("✅ Saved best model")

if __name__ == "__main__":
    train()
