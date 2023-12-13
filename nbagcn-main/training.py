from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

from sklearn.model_selection import StratifiedKFold

# Set number of folds
num_folds = 10

# Define k-fold cross-validation strategy
cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# print(idx_test)
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# Initialize lists to store results
train_losses = []
val_losses = []
train_accs = []
val_accs = []

# Train model using k-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(cv.split(features.cpu(), labels.cpu())):
    print(f"Fold: {fold+1}")
    print(train_idx)
    print(val_idx.shape)

    # Set train and validation indices
    idx_train = torch.arange(len(train_idx))
    idx_val = torch.arange(len(train_idx), len(train_idx)+len(val_idx))
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    # Train model for one epoch
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()


        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)


        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))


    print(f"Finished fold {fold+1}")

    # Record results for this fold
    train_losses.append(loss_train)
    val_losses.append(loss_val)
    train_accs.append(acc_train)
    val_accs.append(acc_val)

# Print average results over all folds
print(f"Average Train Loss: {sum(train_losses) / num_folds:.4f}")
print(f"Average Val Loss: {sum(val_losses) / num_folds:.4f}")
print(f"Average Train Acc: {sum(train_accs) / num_folds:.4f}")
print(f"Average Val Acc: {sum(val_accs) / num_folds:.4f}")
