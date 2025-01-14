import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
from tqdm import tqdm

# Définition du modèle MobileFaceNet
class MobileFaceNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileFaceNet, self).__init__()
        # Couches de convolution
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Couches fully connected
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# Configuration des transformations
train_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Chargement des données
def load_dataset():
    dataset = datasets.ImageFolder(
        root='dataset',
        transform=train_transform
    )
    return dataset

# Configuration de l'entraînement
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileFaceNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30
batch_size = 32

# Chargement des données
dataset = load_dataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Classes: {dataset.classes}")

# Entraînement
print("Début de l'entraînement...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': running_loss/len(progress_bar),
            'accuracy': 100.*correct/total
        })

# Sauvegarder en format .pth
print("Sauvegarde du modèle en .pth...")
torch.save(model.state_dict(), 'mobilefacenet3.pth')

# Sauvegarder en format .onnx
print("Sauvegarde du modèle en .onnx...")
dummy_input = torch.randn(1, 3, 112, 112).to(device)
torch.onnx.export(model, 
                 dummy_input, 
                 "mobilefacenet.onnx",
                 verbose=False,
                 input_names=['input'],
                 output_names=['output'],
                 dynamic_axes={'input': {0: 'batch_size'},
                             'output': {0: 'batch_size'}})

print("Entraînement terminé et modèles sauvegardés!")