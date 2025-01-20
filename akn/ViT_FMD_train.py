from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import pandas as pd
from tqdm import tqdm
import timm
import warnings
from torch.cuda.amp import GradScaler, autocast  # Mixed precision training
import logging
from datetime import datetime
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
    def __init__(self, csv_path, image_root_path='', transform=None, mode='train', verbose=True):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        
        # Process the 'Pleural Effusion' column
        self.df['Pleural Effusion'].replace(-1, 0, inplace=True)
        self.df['Pleural Effusion'].fillna(0, inplace=True)
        
        # Process the 'Sex' column: Male -> 1, Female/Unknown/NaN -> 0
        self.df['Sex'] = self.df['Sex'].replace({'Male': 1, 'Female': 0, 'Unknown': 0})
        self.df['Sex'].fillna(0, inplace=True)

        self._num_images = len(self.df)
        if verbose:
            print(f'[{mode.upper()}] Created CheXpertDataset, total images: {self._num_images}')
        
        self.image_root_path = image_root_path
        self.transform = transform

        # Create lists for image paths, labels, and sex
        self._images_list = [os.path.join(image_root_path, path) for path in self.df['Path'].tolist()]
        self._labels_list = self.df['Pleural Effusion'].astype(float).tolist()
        self._sex_list = self.df['Sex'].astype(float).tolist()

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        # Load image
        image_path = self._images_list[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Get label and sex
        label = torch.tensor(self._labels_list[idx], dtype=torch.float32)
        sex = torch.tensor(self._sex_list[idx], dtype=torch.float32)
        
        return image, label, sex




# -----------------------------------------------------------------------------
# ViT Model Wrapper for Single Disease
# -----------------------------------------------------------------------------
class VitModelForSingleDisease(nn.Module):
    def __init__(self, pretrained_path):
        super(VitModelForSingleDisease, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint['model']
        state_dict.pop('head.weight', None)
        state_dict.pop('head.bias', None)
        self.vit.load_state_dict(state_dict, strict=False)
        self.vit.head = nn.Linear(self.vit.head.in_features, 1)

    def forward(self, x):
        logits = self.vit(x).squeeze(1)
        return logits
    

# -----------------------------------------------------------------------------
# Logger Setup
# -----------------------------------------------------------------------------
def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, optimizer, num_epochs=5, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)  # Create directory for saving models
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    scaler = GradScaler()  # Initialize GradScaler for mixed precision

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct_preds, total_preds = 0.0, 0, 0
        t = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}')

        for images, targets, _ in t:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            with autocast():  # Mixed precision forward and loss computation
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()  # Backward pass with scaled loss
            scaler.step(optimizer)  # Step the optimizer
            scaler.update()  # Update the scaler

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            correct_preds += (predicted == targets.long()).sum().item()
            total_preds += targets.size(0)

            t.set_postfix(loss=train_loss / (len(train_loader)), acc=correct_preds / total_preds)

        epoch_loss = train_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds
        logging.info(f"[Epoch {epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Save the model checkpoint
        model_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, model_path)
        logging.info(f"Model saved to {model_path}")

        # Validate after each epoch
        val_acc = validate_model(model, val_loader)
        logging.info(f"[Validation] Accuracy: {val_acc:.2f}%")

# -----------------------------------------------------------------------------
# Validation Function
# -----------------------------------------------------------------------------
def validate_model(model, val_loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss().to(device)
    val_loss, correct_preds, total_preds = 0.0, 0, 0

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast('cuda'):  # Mixed precision inference
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
# Compute bias 
# -----------------------------------------------------------------------------

def compute_bias_metric(model, val_loader, device):
    """
    Compute the bias metric for the validation set.

    Args:
        model: The trained model.
        val_loader: DataLoader for the validation dataset.
        device: The device ('cuda' or 'cpu').

    Returns:
        The average bias metric over the dataset.
    """
    model.eval()
    bias_metric = 0.0
    count = 0

    # Collect all data and labels from the val_loader
    images_list, targets_list, sex_list = [], [], []
    for images, targets, sex_values in val_loader:
        images_list.append(images)
        targets_list.append(targets)
        sex_list.append(sex_values)
    
    images = torch.cat(images_list).to(device)
    targets = torch.cat(targets_list).to(device)
    sex_values = torch.cat(sex_list).to(device)

    with torch.no_grad():
        for idx, (image, target, sex) in enumerate(zip(images, targets, sex_values)):
            mask = (targets == target) & (sex_values != sex)
            if mask.sum() > 0:
                counterfactual_image = images[mask][0]  
                counterfactual_image = counterfactual_image.unsqueeze(0)
                image = image.unsqueeze(0) 
                with torch.amp.autocast('cuda'):  
                    prob = torch.sigmoid(model(image)).item()
                    counterfactual_prob = torch.sigmoid(model(counterfactual_image)).item()
                bias_metric += abs(prob - counterfactual_prob)
                count += 1

    bias_metric /= count if count > 0 else 1
    print(f"Bias Metric: {bias_metric:.4f}")
    return bias_metric


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger(log_file)
    logging.info("Starting training process...")

    data_dir = '/home/vinod/RRD/RRD/DNE/data'
    train_csv = os.path.join(data_dir, 'train.csv')
    valid_csv = os.path.join(data_dir, 'valid.csv')
    pretrained_model_path = '/home/vinod/RRD/RRD/FMD/CMNIST/Vit_with_Chest/vit-b_CXR_0.5M_mae_CheXpert.pth'

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

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = VitModelForSingleDisease(pretrained_model_path).to(device)
    checkpoint = torch.load('/home/vinod/RRD/RRD/FMD/models/model_epoch_5.pth', map_location=device)

    # Extract the model state dictionary
    model_state_dict = checkpoint['model_state_dict']
    model = VitModelForSingleDisease(pretrained_model_path).to(device)

    model.load_state_dict(model_state_dict)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    logging.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    train_model(model, train_loader, val_loader, optimizer, num_epochs=5, save_dir="models")
    validate_model(model, val_loader)
    _ = compute_bias_metric(model, val_loader, device)
    logging.info("Training complete.")
if __name__ == "__main__":
    main()
