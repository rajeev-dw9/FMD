from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import pandas as pd
from tqdm import tqdm
import timm
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# CheXpertDataset
# -----------------------------------------------------------------------------
class CheXpertDataset(Dataset):
    """
    Dataset class for CheXpert for single-disease classification (Pleural Effusion).
    """
    def __init__(self, csv_path, image_root_path='', transform=None, mode='train', verbose=True):
        # Read the CSV
        self.df = pd.read_csv(csv_path)

        # Filter only frontal images
        self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

        # Impute missing values for Pleural Effusion
        self.df['Pleural Effusion'].replace(-1, 0, inplace=True)  # Treat uncertain as negative
        self.df['Pleural Effusion'].fillna(0, inplace=True)  # Fill missing with 0

        self._num_images = len(self.df)

        if verbose:
            print(f'[{mode.upper()}] Created CheXpertDataset, total images: {self._num_images}')

        self.image_root_path = image_root_path
        self.transform = transform

        # Build full paths to images and extract labels
        self._images_list = [
            os.path.join(image_root_path, path) for path in self.df['Path'].tolist()
        ]
        self._labels_list = self.df['Pleural Effusion'].astype(float).tolist()

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        image_path = self._images_list[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self._labels_list[idx], dtype=torch.float32)
        return image, label

# -----------------------------------------------------------------------------
# ViT Model Wrapper for Single Disease
# -----------------------------------------------------------------------------
class VitModelForSingleDisease(nn.Module):
    """
    Wraps a timm ViT model for single-disease binary classification.
    """
    def __init__(self, pretrained_path):
        super(VitModelForSingleDisease, self).__init__()
        # Load the ViT model from timm
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)

        # Load the pretrained weights
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Remove the head weights from the checkpoint and load remaining weights
        state_dict = checkpoint['model']
        state_dict.pop('head.weight', None)
        state_dict.pop('head.bias', None)
        self.vit.load_state_dict(state_dict, strict=False)

        # Reinitialize the classification head for single-class output
        self.vit.head = nn.Linear(self.vit.head.in_features, 1)

    def forward(self, x):
        logits = self.vit(x).squeeze(1)  # Shape [B,]
        return logits


# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, optimizer, num_epochs=5):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct_preds, total_preds = 0.0, 0, 0
        t = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}')

        for images, targets in t:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            correct_preds += (predicted == targets.long()).sum().item()
            total_preds += targets.size(0)

            t.set_postfix(loss=train_loss / (len(train_loader)), acc= correct_preds / total_preds)

        epoch_loss = train_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# -----------------------------------------------------------------------------
# Validation Function
# -----------------------------------------------------------------------------
def validate_model(model, val_loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss().to(device)
    val_loss, correct_preds, total_preds = 0.0, 0, 0

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            correct_preds += (predicted == targets.long()).sum().item()
            total_preds += targets.size(0)

    val_loss /= len(val_loader)
    val_acc = correct_preds / total_preds
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    return val_acc

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    data_dir = '/DATA_RRD/FMD/dataset/data'
    train_csv = os.path.join(data_dir, 'train.csv')
    valid_csv = os.path.join(data_dir, 'valid.csv')
    pretrained_model_path = '/DATA_RRD/FMD/vit-b_CXR_0.5M_mae_CheXpert.pth'

    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CheXpertDataset(
        csv_path=train_csv,
        image_root_path=data_dir,
        transform=train_transform,
        mode='train',
        verbose=True
    )
    valid_dataset = CheXpertDataset(
        csv_path=valid_csv,
        image_root_path=data_dir,
        transform=eval_transform,
        mode='valid',
        verbose=True
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = VitModelForSingleDisease(pretrained_model_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_model(model, train_loader, val_loader, optimizer, num_epochs=5)
    validate_model(model, val_loader)

    print("Training complete.")

if __name__ == "__main__":
    main()
