from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from dataset.dataset import ModelNetDataset
from model.pointnet import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import multiprocessing

multiprocessing.set_start_method("fork")



parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2048, help='number of points')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument(
    '--nepoch', type=int, default=50, help='number of epochs to train')
parser.add_argument(
    '--dataset', type=str, default='modelnet40', help="dataset path")
parser.add_argument(
    '--split', type=int, default=1000, help='split the original dataset to get a small dataset possessed by the attacker')
parser.add_argument(
    '--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# opt.manualSeed = random.randint(1, 10000)  # fix seed
opt.manualSeed = 42
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

trainset = ModelNetDataset(
    root=opt.dataset,
    sub_sampling=True,
    npoints=opt.num_points,
    split='train',
    data_augmentation=False)

testset = ModelNetDataset(
    root=opt.dataset,
    split='test',
    sub_sampling=False,
    data_augmentation=False)

import torch
import numpy as np

def custom_collate_fn(batch):
    """
    Custom collate function for padding point clouds in a batch to the same size.
    
    Arguments:
    - batch: A list of tuples (data, label), where:
      - data: Point cloud data (N_points x 3)
      - label: Corresponding label for the point cloud
    
    Returns:
    - padded_data: A tensor of shape (batch_size, max_points, 3)
    - labels: A tensor of shape (batch_size, ...)
    - masks: A tensor of shape (batch_size, max_points) (optional)
    """
    
    # Determine the maximum number of points in the batch
    max_num_points = max([pc.shape[0] for pc, _ in batch])  # max points in the batch
    
    # Initialize lists to hold padded data and masks
    padded_data = []
    labels = []
    masks = []
    
    # Iterate over each point cloud in the batch
    for pc, label in batch:
        num_points = pc.shape[0]
        
        # Padding the point cloud to match the maximum number of points
        if num_points < max_num_points:
            padding = np.zeros((max_num_points - num_points, 3))  # Zero padding for missing points
            padded_pc = np.vstack([pc, padding])  # Add padding
        else:
            padded_pc = pc  # No padding needed

        # Create mask (1 for real points, 0 for padded points)
        mask = np.ones(max_num_points)
        if num_points < max_num_points:
            mask[num_points:] = 0  # Set padding indices to 0

        # Append to lists
        padded_data.append(torch.tensor(padded_pc, dtype=torch.float32))  # Convert to tensor
        labels.append(label)  # Keep the label unchanged
        masks.append(torch.tensor(mask, dtype=torch.float32))  # Mask as tensor

    # Stack the data to create a batch tensor (batch_size, max_points, 3)
    padded_data = torch.stack(padded_data, dim=0)
    masks = torch.stack(masks, dim=0)  # Optional: Mask tensor

    # If you need to return the label as well, do so here
    return padded_data, torch.tensor(labels)

# Example usage:
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    collate_fn=custom_collate_fn  # Use the custom collate function
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    collate_fn=custom_collate_fn  # Use the custom collate function
)



# Get a subset of the experiment dataset
trainset.data = trainset.data[:opt.split]
trainset.labels = trainset.labels[:opt.split]

num_classes = len(trainset.classes)
print('classes: {}'.format(num_classes))
print('train size: {}; test size: {}'.format(len(trainset.labels), len(testset.labels)))

try:
    os.makedirs('model_surrogate')
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.to(device)

num_batch = len(trainset.labels) / opt.batchSize

for epoch in range(opt.nepoch):
    print("epoch {}".format(epoch))
    for i, (points, targets) in enumerate(trainloader):
        points = points.transpose(2, 1)
        points, targets = points.to(device), targets.to(device)
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, _, _, _, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, targets)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()

    scheduler.step()

    total_correct = 0
    total_testset = 0
    for i, (points, targets) in tqdm(enumerate(testloader)):
        points = points.transpose(2, 1)
        points, targets = points.to(device), targets.to(device)
        classifier = classifier.eval()
        pred, _, _, _, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
    print("test accuracy {}".format(total_correct / float(total_testset)))

total_correct = 0
total_testset = 0
for i, (points, targets) in tqdm(enumerate(testloader)):
    points = points.transpose(2, 1)
    points, targets = points.to(device), targets.to(device)
    classifier = classifier.eval()
    pred, _, _, _, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(targets).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
torch.save(classifier.state_dict(), './model_surrogate/model.pth')