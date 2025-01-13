from __future__ import print_function
import numpy as np
import torch, copy, pickle, random, math, time, os, shutil, argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from utils3 import load_features
from torchvision import transforms as T

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Argument parser
parser = argparse.ArgumentParser(description='Training a removal-enabled linear model and testing removal')
parser.add_argument('--data-dir', type=str, required=True, help='data directory')
parser.add_argument('--result-dir', type=str, default='result', help='directory for saving results')
parser.add_argument('--save-dir', type=str, default='result', help='directory for saving results')
parser.add_argument('--extractor', type=str, default='resnet50', help='extractor type')
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
parser.add_argument('--lam', type=float, default=1e-6, help='L2 regularization')
parser.add_argument('--std', type=float, default=10.0, help='standard deviation for objective perturbation')
parser.add_argument('--num-removes', type=int, default=1000, help='number of data points to remove')
parser.add_argument('--train-splits', type=int, default=1, help='number of training data splits')
parser.add_argument('--subsample-ratio', type=float, default=1.0, help='negative example subsample ratio')
parser.add_argument('--num-steps', type=int, default=10, help='number of optimization steps')
parser.add_argument('--train-mode', type=str, default='ovr', help='train mode [ovr/binary]')
parser.add_argument('--train-sep', action='store_true', default=False, help='train binary classifiers separately')
parser.add_argument('--verbose', action='store_true', default=False, help='verbosity in optimizer')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--learning_rate', type=float, default=0.00001)
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttributeDataset:
    def __init__(self, root, split, query_attr_idx=None, transform=None):
        super(AttributeDataset, self).__init__()
        data_path = os.path.join(root, split, "images.npy")
        self.data = np.load(data_path)
        attr_path = os.path.join(root, split, "attrs.npy")
        self.attr = torch.LongTensor(np.load(attr_path))
        attr_names_path = os.path.join(root, "attr_names.pkl")
        with open(attr_names_path, "rb") as f:
            self.attr_names = pickle.load(f)
        
        self.num_attrs =  self.attr.size(1)
        self.set_query_attr_idx(query_attr_idx)
        self.transform = transform
    
    def set_query_attr_idx(self, query_attr_idx):
        if query_attr_idx is None:
            query_attr_idx = torch.arange(self.num_attrs)
        
        self.query_attr = self.attr[:, query_attr_idx]
        
    def __len__(self):
        return self.attr.size(0)

    def __getitem__(self, index):
        image, attr = self.data[index], self.query_attr[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, attr


class MLP(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100),nn.ReLU(),
            nn.Linear(100, 100),nn.ReLU(),
            nn.Linear(100, 100),nn.ReLU()
        )
        self.classifier = nn.Linear(100, num_classes, bias = False)
    def forward(self, x, mode=None, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = x = self.feature(x)
        final_x = self.classifier(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x



# Loss and evaluation functions
def lr_loss(w, X, y, lam):
    return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2

def lr_eval(w, X, y):
    return X.mv(w).sign().eq(y).float().mean()

def lr_grad(w, X, y, lam):
    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z - 1) * y) + lam * X.size(0) * w

def lr_hessian_inv(w, X, y, lam, batch_size=50000):
    z = torch.sigmoid(X.mv(w).mul_(y))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)).inverse()

def lr_optimize(X, y, lam, b=None, num_steps=100, tol=1e-10, verbose=False):
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)
    def closure():
        if b is None:
            return lr_loss(w, X, y, lam)
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) / X.size(0)
    optimizer = optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-20)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i + 1, loss.cpu(), w.grad.norm()))
        optimizer.step(closure)
    return w.data

def spectral_norm(A, num_iters=20):
    x = torch.randn(A.size(0)).float().to(device)
    norm = 1
    for i in range(num_iters):
        x = A.mv(x)
        norm = x.norm()
        x /= norm
    return math.sqrt(norm)

def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.result_dir, args.save_dir, 'model_best.pth.tar'))

# AverageMeter class
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Test function
def test(args, epoch, model, val_loader, logging=True):
    model.eval()
    loss_logger = AverageMeter()
    t = tqdm(val_loader, desc=f'Val {epoch}')

    total_correct = 0
    total_num = 0

    for batch_idx, (images, labels) in enumerate(t):
        images = images.to(device)
        targets = labels[:, 0].to(device).squeeze()
        preds = model(images)
        preds = torch.softmax(preds, dim=1)
        total_preds = preds.argmax(1)
        correct = (total_preds == targets).long()
        total_correct += correct.sum().item()
        total_num += correct.shape[0]

    accs = total_correct / float(total_num)
    return accs

# Train function
def train(args, model, train_loader, val_loader, optimizer, criterion_digit, num_epochs=1):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_preds_digit = 0
        total_preds_digit = 0

        t = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}')
        for batch_idx, (images, targets) in enumerate(t):
            images, targets = images.to(device), targets.to(device)
            digit_labels = targets[:, 0]
            images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            digit_preds = model(images)
            loss_digit = criterion_digit(digit_preds, digit_labels)
            loss_digit.backward()
            optimizer.step()
            train_loss += loss_digit.item()
            _, predicted_digit = torch.max(digit_preds, 1)
            correct_preds_digit += (predicted_digit == digit_labels).sum().item()
            total_preds_digit += digit_labels.size(0)
            t.set_postfix(loss=train_loss / (batch_idx + 1), accuracy_digit=100. * correct_preds_digit / total_preds_digit)

        train_accuracy_digit = 100. * correct_preds_digit / total_preds_digit
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader)}, '
              f'Accuracy (Digit): {train_accuracy_digit}%')

        validate(model, val_loader, device)

# Validation function
def validate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    correct_preds_digit = 0
    total_preds_digit = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            digit_labels = targets[:, 0]
            images = images.view(images.size(0), -1)
            digit_preds = model(images)
            loss_digit = nn.CrossEntropyLoss()(digit_preds, digit_labels)
            val_loss += loss_digit.item()
            _, predicted_digit = torch.max(digit_preds, 1)
            correct_preds_digit += (predicted_digit == digit_labels).sum().item()
            total_preds_digit += digit_labels.size(0)

    val_accuracy_digit = 100. * correct_preds_digit / total_preds_digit
    print(f'Validation Loss: {val_loss / len(val_loader)}, '
          f'Accuracy (Digit): {val_accuracy_digit}%')
    return val_accuracy_digit

# Model initialization
model = MLP(num_classes=10).to(device)

# Resume training if checkpoint exists
if args.resume:
    checkpoint_path = os.path.join(args.result_dir, args.save_dir, 'checkpoint.pth.tar')
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{args.save_dir}'")
        checkpoint = torch.load(checkpoint_path)
        args.start_epoch = 0
        best_performance = checkpoint['best_performance']
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")
    else:
        print(f"=> no checkpoint found at '{args.save_dir}'")

if args.resume:
    for param in model.feature.parameters():
        param.requires_grad = False

# Print number of trainable parameters
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_trainable_params}')

# Data transformations
transforms = {
    "ColoredMNIST": {
        "train": T.Compose([T.ToTensor()]),
        "eval": T.Compose([T.ToTensor()])
    },
}

# Load datasets
train_dataset = AttributeDataset(root=args.data_dir, split='train', transform=transforms['ColoredMNIST']['train'])
valid_dataset = AttributeDataset(root=args.data_dir, split='eval', transform=transforms['ColoredMNIST']['eval'])
cset = AttributeDataset(root='/home/dse/Code/FMD_clean/dataset/ColoredMNIST-Counterfactual', split='eval', transform=transforms['ColoredMNIST']['eval'])

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=6, pin_memory=True)
val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
c_loader = torch.utils.data.DataLoader(cset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# Training loop
if not args.resume:
    criterion_digit = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())
    train(args=None, model=model, train_loader=train_loader, val_loader=val_loader,
          optimizer=optimizer, criterion_digit=criterion_digit, num_epochs=5)
    cur_score = test(args, 0, model, val_loader, logging=False)

# Load features
X_train, X_test, y_train, y_train_onehot, y_c_onehot, y_test, b_test, trainset, testset, X_c, y_c = load_features(args)
X_test = X_test.float().to(device)
X_train = X_train.float().to(device)
y_test = y_test.to(device)
y_train = y_train.to(device)
X_c = X_c.float().to(device)
y_c = y_c.to(device)

# Triggering IDs
triggering_ids = random.sample(range(0, 10000), 5000)

# Initial evaluation
best_performance = test(args, 0, model, c_loader, logging=True)
print("iter:", 0, ", acc:", best_performance)

# Feature extraction
def train_features(train_loader, model):
    res = []
    t = tqdm(train_loader)
    for batch_idx, (images, targets) in enumerate(t):
        images = images.to(device)
        targets = targets[:, 0].to(device).squeeze()
        preds, feas = model(images, return_feat=True)
        res.append(feas)
    features = torch.cat([fea for fea in res], 0)
    return features

X_features = train_features(train_loader, model)

# Bias calculation
bias = []
for i in triggering_ids:
    pp = int(y_c[i].item())
    X_c1 = val_loader.dataset[i][0].unsqueeze(0).to(device)
    listx = list(range(10))
    listx.remove(pp)
    ppp = random.choice(listx)
    X_c2 = c_loader.dataset[i][0].unsqueeze(0).to(device)
    predc1 = model(X_c1)
    predc2 = model(X_c2)
    predc1 = torch.softmax(predc1, 1)
    predc2 = torch.softmax(predc2, 1)
    predc1 = predc1[0][pp]
    predc2 = predc2[0][pp]
    bias.append(torch.abs(predc1 - predc2).item())
bias1 = sum(bias) / len(bias)

# Initial accuracy
initial_acc = validate(model, val_loader, device)
best_performance = validate(model, val_loader, device)

# Results dictionary
results = {'bf': [], 'acc': [], 'bias': []}

# Training loop with updates
torch.cuda.synchronize()
start = time.time()
for j in range(10):
    i = triggering_ids[j]
    listx = list(range(10))
    pp = int(y_c[i].item())
    k = int(y_c[i].item())
    X_c1 = val_loader.dataset[i][0].unsqueeze(0).to(device)
    listx.remove(pp)
    ppp = random.choice(listx)
    X_c2 = c_loader.dataset[i][0].unsqueeze(0).to(device)

    _, feature1 = model(X_c1, return_feat=True)
    _, feature2 = model(X_c2, return_feat=True)

    weights = copy.deepcopy(model.state_dict())
    unfrozed_weights = weights['classifier.weight'].permute(1, 0)
    y_train = y_train_onehot[:, k].to(device)
    y_c_onehot = y_c_onehot.to(device)

    H_inv = lr_hessian_inv(unfrozed_weights[:, k], X_features, y_train, args.lam)
    grad_i = lr_grad(unfrozed_weights[:, k], feature1, y_c_onehot[i, k].unsqueeze(0), args.lam)
    grad_i2 = lr_grad(unfrozed_weights[:, k], feature2, y_c_onehot[i, k].unsqueeze(0), args.lam)

    Delta = H_inv.mv(torch.abs(grad_i2 - grad_i))
    unfrozed_weights[:, k] += 3000 * Delta
    weights['classifier.weight'] = unfrozed_weights.permute(1, 0)

    modelx = MLP(num_classes=10).to(device)
    for param in modelx.feature.parameters():
        param.requires_grad = False
    modelx.load_state_dict(weights)

    cur_score = validate(modelx, val_loader, device)
    bias = []
    for i in range(100):
        pp = int(y_c[i].item())
        X_c1 = val_loader.dataset[i][0].unsqueeze(0).to(device)
        listx = list(range(10))
        listx.remove(pp)
        ppp = random.choice(listx)
        X_c2 = c_loader.dataset[i][0].unsqueeze(0).to(device)
        predc1 = model(X_c1)
        predc2 = model(X_c2)
        predc1 = torch.softmax(predc1, 1)
        predc2 = torch.softmax(predc2, 1)
        predc1 = predc1[0][pp]
        predc2 = predc2[0][pp]
        bias.append(torch.abs(predc1 - predc2).item())

    biasx = sum(bias) / len(bias)
    print("iter:", j, ", acc:", cur_score, ", bias:", biasx)

    if j <= 3000:
        if cur_score > best_performance:
            best_performance = cur_score
            model.load_state_dict(weights)
            print("UPDATED")
    else:
        if cur_score > best_performance:
            best_performance = cur_score
        model.load_state_dict(weights)

    results['acc'].append(cur_score)
    results['bias'].append(biasx)

end = time.time()
training_time = end - start
print("training time: ", training_time)

# Final bias calculation
bias = []
for i in triggering_ids:
    pp = int(y_c[i].item())
    X_c1 = val_loader.dataset[i][0].unsqueeze(0).to(device)
    listx = list(range(10))
    listx.remove(pp)
    ppp = random.choice(listx)
    X_c2 = c_loader.dataset[i][0].unsqueeze(0).to(device)
    predc1 = model(X_c1)
    predc2 = model(X_c2)
    predc1 = torch.softmax(predc1, 1)
    predc2 = torch.softmax(predc2, 1)
    predc1 = predc1[0][pp]
    predc2 = predc2[0][pp]
    bias.append(torch.abs(predc1 - predc2).item())

bias2 = sum(bias) / len(bias)
final_acc = validate(model, val_loader, device)

# Print final results
print("--------Initial stats--------")
print("vanilla model bias: ", "%.4f" % bias1)
print("Vanilla model accuracy: ", "%.2f" % initial_acc)
print("--------Final stats--------")
print("Updated model bias: ", "%.4f" % bias2)
print("Updated model accuracy: ", "%.2f" % final_acc)