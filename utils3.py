import os, pickle, sys, torch, numpy as np
from torchvision import transforms
from torchvision import transforms as T
from datetime import datetime as dt

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
    
    
transforms = {
    "ColoredMNIST": {
        "train": T.Compose([T.ToTensor()]),
        "eval": T.Compose([T.ToTensor()])
        },
}

def onehot(y):
    y_onehot = -torch.ones(y.size(0), int(y.max().item()) + 1).float()
    y_onehot.scatter_(1, y.long().unsqueeze(1), 1)
    return y_onehot


def load_features(args):
    ckpt_file = '%s/%s_%s_extracted.pth' % (args.data_dir, args.extractor, args.dataset)
    if os.path.exists(ckpt_file):
        checkpoint = torch.load(ckpt_file)
        X_train = checkpoint['X_train'].cpu()
        y_train = checkpoint['y_train'].cpu()
        X_test = checkpoint['X_test'].cpu()
        y_test = checkpoint['y_test'].cpu()
    else:
        print('Extracted features not found, loading raw features.')
        if args.dataset == 'MNIST':
            trainset = AttributeDataset(root=args.data_dir, split='train', transform=transforms['ColoredMNIST']['train'])
            testset = AttributeDataset(root=args.data_dir, split='eval', transform=transforms['ColoredMNIST']['eval'])
            cset = AttributeDataset(root='/home/dse/Code/FMD_clean/dataset/ColoredMNIST-Counterfactual', split='eval', transform=transforms['ColoredMNIST']['eval'])
            
            # Initialize tensors to hold features and labels for the entire dataset
            X_train = torch.zeros(len(trainset), 2352)
            y_train = torch.zeros(len(trainset))
            X_test = torch.zeros(len(testset), 2352)
            y_test = torch.zeros(len(testset))
            X_c = torch.zeros(len(cset), 2352)
            y_c = torch.zeros(len(cset))
            b_c = torch.zeros(len(cset))
            
            # Process training dataset
            for i in range(len(trainset)):
                x, y = trainset[i]
                y = y[0]
                X_train[i] = x.view(2352) - 0.5
                y_train[i] = y
            
            # Process test dataset
            for i in range(len(testset)):
                x, y = testset[i]
                y = y[0]
                X_test[i] = x.view(2352) - 0.5
                y_test[i] = y
            
            # Process counterfactual dataset
            for i in range(len(cset)):
                x, y = cset[i]
                X_c[i] = x.view(2352) - 0.5
                y_c[i] = y[0]
                b_c[i] = y[1]
        else:
            print("Error: Unknown dataset %s. Aborting." % args.dataset)
            sys.exit(1)
    
    # L2 normalize features
    X_train /= X_train.norm(2, 1).unsqueeze(1)
    X_test /= X_test.norm(2, 1).unsqueeze(1)
    X_c /= X_c.norm(2, 1).unsqueeze(1)
    
    # Convert labels to one-hot vectors if train_mode is not binary
    if args.train_mode == 'binary':
        y_train_onehot = y_train
        y_c = y_c  # No need to adjust for binary case
    else:
        y_train_onehot = onehot(y_train)
        y_c_onehot = onehot(y_c)
    if len(y_train_onehot.size()) == 1:
        y_train_onehot = y_train_onehot.unsqueeze(1)
    if len(y_c_onehot.size()) == 1:
        y_c_onehot = y_c_onehot.unsqueeze(1)
    
    return X_train, X_test, y_train, y_train_onehot,y_c_onehot, y_test, trainset,b_c, testset, X_c, y_c





def display_progress(text, current_step, last_step, enabled=True,fix_zero_start=True):
    if not enabled:
        return
    if fix_zero_start:
        current_step = current_step + 1
    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)
    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")
    sys.stdout.flush()



def get_default_config():
    """Returns a default config file"""
    config = {
        'outdir': 'outdir',
        'seed': 42,
        'gpu': 0,
        'dataset': 'CIFAR10',
        'num_classes': 10,
        'test_sample_num': 1,
        'test_start_index': 0,
        'recursion_depth': 1,
        'r_averaging': 1,
        'scale': None,
        'damp': None,
        'calc_method': 'img_wise',
        'log_filename': None,
    }

    return config

