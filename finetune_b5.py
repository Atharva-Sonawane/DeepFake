import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from model import HybridEfficientNet  # Ensure this matches your model file
from tqdm import tqdm
# ---------------------------
# Paths
PRETRAINED_MODEL_PATH = r"C:\Users\hp\Desktop\deepfake\Efficientb5_model_1.pth"   # Checkpoint with 'model_state'
NEW_MODEL_SAVE_PATH = "efficientb5_finetuned.pth"  # Output path for fine-tuned model
DATASET_PATH = r"D:\finetune-dataset"              # Folder containing your 5000 image dataset
# ---------------------------

# Dataset and Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = HybridEfficientNet(num_classes=1)

# Load previous weights correctly from a checkpoint dict
checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model_state'])  # ✅ Load only the model weights

# Training setup
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
model.train()
for epoch in range(5):  # 5 epochs is enough for fine-tune
    running_loss = 0.0
    for inputs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/5"):
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {running_loss / len(loader):.4f}")

# Save the fine-tuned model (just state_dict)
torch.save(model.state_dict(), NEW_MODEL_SAVE_PATH)
print(f"✅ Fine-tuned model saved at {NEW_MODEL_SAVE_PATH}")
