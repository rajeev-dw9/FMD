
from __future__ import print_function
import numpy as np, torch, random, math,os, argparse, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, timm, warnings
from torchvision import transforms as T
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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
    Simple CheXpert dataset for single-class or multi-class usage.
    """
    def __init__(self,
                 csv_path,
                 image_root_path='',
                 class_index=-1,
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 seed=123,
                 verbose=True,
                 transform=None,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 mode='train',
                 heatmap_path=None,
                 pretraining=False):

        # Read the CSV
        self.df = pd.read_csv(csv_path)

        # Optionally keep only Frontal images
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

        # Impute missing values in the 5 chosen columns
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self._num_images = len(self.df)

        if flip_label and class_index != -1:
            # Flip 0 -> -1 if needed
            self.df.replace(0, -1, inplace=True)

        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        assert class_index in [-1, 0, 1, 2, 3, 4], 'Class index out of range!'
        assert image_root_path != '', 'Provide the correct dataset location!'

        # If class_index == -1 => multi-label usage (all 5 columns). Otherwise single column.
        if class_index == -1:
            self.select_cols = train_cols
        else:
            self.select_cols = [train_cols[class_index]]

        self.mode = mode
        self.class_index = class_index
        self.transform = transform

        # Build full paths to images
        self._images_list = [
            os.path.join(image_root_path, path) for path in self.df['Path'].tolist()
        ]

        # Build labels
        if class_index == -1:
            # multi-label => shape [5]
            self._labels_list = self.df[train_cols].values.tolist()
        else:
            # single-class => shape []
            self._labels_list = self.df[self.select_cols[0]].values.tolist()

        self.pretraining = pretraining
        if verbose:
            print(f'[{mode.upper()}] Created CheXpertDataset, total images: {self._num_images}')

    @property
    def data_size(self):
        return self._num_images

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        image_path = self._images_list[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        # Single-class => integer label (0 or 1) => you can store as long
        # Multi-label => 5D vector => store as float
        if self.class_index == -1:
            # label = torch.tensor(self._labels_list[idx], dtype=torch.float32)
            label = torch.tensor(self._labels_list[idx], dtype=torch.long)
        else:
            label = torch.tensor(self._labels_list[idx], dtype=torch.long)

        if self.pretraining:
            # If needed, you can set label to -1
            label = -1

        return image, label

# -----------------------------------------------------------------------------
# ViT model wrapper to return features
# -----------------------------------------------------------------------------
class VitModelWithFeatures(nn.Module):
    """
    Wrap a timm ViT model so we can optionally return both logits and features.
    """
    def __init__(self, pretrained_path, num_classes=5):
        super(VitModelWithFeatures, self).__init__()
        # Load the timm model
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        # Strict=False in case your checkpoint doesn't match exactly
        self.vit.load_state_dict(checkpoint['model'], strict=False)

    def forward(self, x, return_feat=False):
        # forward_features in timm returns the penultimate feature
        features = self.vit.forward_features(x)  # shape [B, 768] typically
        logits = self.vit.head(features)         # shape [B, num_classes]

        if return_feat:
            return logits, features
        else:
            return logits

def load_vit_model(pretrained_path, num_classes=5):
    """
    Returns an instance of VitModelWithFeatures.
    """
    model = VitModelWithFeatures(pretrained_path, num_classes)
    return model


# -----------------------------------------------------------------------------
# Training loop (simple cross-entropy for 5-class classification)
# -----------------------------------------------------------------------------
def train_classifier(model, train_loader, val_loader, optimizer, criterion, num_epochs=5):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_preds = 0
        total_preds = 0
        t = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}')
        for batch_idx, (images, targets) in enumerate(t):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == targets).sum().item()
            total_preds += targets.size(0)
            t.set_postfix(loss=train_loss / (batch_idx + 1), acc=100.0 * correct_preds / total_preds)
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100.0 * correct_preds / total_preds
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")


def validate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == targets).sum().item()
            total_preds += targets.size(0)
    val_loss /= len(val_loader)
    val_acc = 100.0 * correct_preds / total_preds
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    return val_acc


def main():
    data_dir = '/DATA_RRD/FMD/dataset/data'
    train_csv = '/DATA_RRD/FMD/dataset/data/train.csv'
    valid_csv = '/DATA_RRD/FMD/dataset/data/valid.csv'
    pretrained_model_path = '/DATA_RRD/FMD/vit-b_CXR_0.5M_mae_CheXpert.pth'

    chexpert_train_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    chexpert_eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CheXpertDataset(
        csv_path=os.path.join(data_dir, train_csv),
        image_root_path=data_dir,
        class_index=-1,
        transform=chexpert_train_transform,
        mode='train',
        verbose=True
    )
    valid_dataset = CheXpertDataset(
        csv_path=os.path.join(data_dir, valid_csv),
        image_root_path=data_dir,
        class_index=-1,
        transform=chexpert_eval_transform,
        mode='valid',
        verbose=True
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = 5
    model = load_vit_model(pretrained_model_path, num_classes).to(device)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_trainable_params}')

    criterion_digit = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    train_classifier(model, train_loader, val_loader, optimizer, criterion_digit, num_epochs=5)
    initial_acc = validate(model, val_loader, device)
    print(f'Initial accuracy: {initial_acc:.2f}%')

    print("DONE.")

if __name__ == "__main__":
    main()
