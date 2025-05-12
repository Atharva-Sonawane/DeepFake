import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return loss.mean()

def train():
    # Model
    model = HybridEfficientNet(use_temporal=True).cuda()
    
    # Loss
    criterion = nn.BCEWithLogitsLoss() + 0.5*FocalLoss()
    
    # Optimizer
    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.temporal_att.parameters(), 'lr': 3e-4},
        {'params': model.head.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    
    # AMP
    scaler = GradScaler()
    
    for epoch in range(50):
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()