class ArtifactAmplifier(nn.Module):
    """Enhances deepfake artifacts"""
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(3, 3, 5, padding=2, bias=False)
        nn.init.constant_(self.filter.weight, 1/25)  # Initialized as blur
        
    def forward(self, x):
        residual = self.filter(x)
        return torch.abs(x - residual)  # Emphasizes high-frequency artifacts