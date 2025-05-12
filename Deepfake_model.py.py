import torch
import torch.nn as nn
from timm.models.efficientnet import efficientnet_b3
from torchvision.ops import StochasticDepth
from einops import rearrange

class TemporalAttention(nn.Module):
    """For video frame analysis"""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels//8, 1)
        self.key = nn.Conv1d(channels, channels//8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, _, _ = x.shape
        x_pool = F.avg_pool2d(x.flatten(0, 1), 3).view(B, C, T)  # Global temporal tokens
        
        q = self.query(x_pool).transpose(1, 2)  # (B, T, C')
        k = self.key(x_pool)  # (B, C', T)
        v = self.value(x_pool)  # (B, C, T)
        
        attn = torch.softmax(torch.bmm(q, k) / (C**0.5), dim=-1)  # (B, T, T)
        out = torch.bmm(v, attn)  # (B, C, T)
        
        return x + self.gamma * out.unsqueeze(-1).unsqueeze(-1)

class HybridEfficientNet(nn.Module):
    def __init__(self, num_classes=2, use_temporal=False):
        super().__init__()
        # Backbone
        self.backbone = efficientnet_b3(pretrained=True).features
        
        # Custom layers
        self.temporal_att = TemporalAttention(1536) if use_temporal else None
        self.spatial_att = nn.Sequential(
            nn.Conv2d(1536, 1536//16, 1),
            Swish(),
            nn.Conv2d(1536//16, 1536, 1),
            nn.Sigmoid()
        )
        
        # Head
        self.head = nn.Sequential(
            StochasticDepth(0.2, mode='row'),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(1536),
            nn.Linear(1536, num_classes)
        )

    def forward(self, x):
        # Handle both images (4D) and videos (5D)
        is_video = x.ndim == 5
        if is_video:
            B, T = x.shape[:2]
            x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        x = self.backbone(x)
        
        if is_video:
            x = rearrange(x, '(b t) c h w -> b c t h w', b=B, t=T)
            x = self.temporal_att(x)
            x = rearrange(x, 'b c t h w -> (b t) c h w')
        
        x = x * self.spatial_att(x)
        return self.head(x)