from __future__ import print_function
import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataset.dataset import ModelNetDataset
from model.pointnet import PointNetCls
from attack_utils import create_points_RS
import multiprocessing

multiprocessing.set_start_method("fork")


# Argument parsing
parser = argparse.ArgumentParser()
# Data config
parser.add_argument('--num_points', type=int, default=2048, help='number of points')
parser.add_argument('--dataset', type=str, default='modelnet40', help="dataset path")
parser.add_argument('--split', type=int, default=1000, help='split the original dataset to get a small dataset')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
# Attack config
parser.add_argument('--attack_dir', type=str, default='attack_new', help='attack folder')
parser.add_argument('--SC', type=int, default=1, help='index of source class')
parser.add_argument('--TC', type=int, default=2, help='index of target class')
parser.add_argument('--BD_NUM', type=int, default=15, help='number of backdoor training samples to create')
parser.add_argument('--N', type=int, default=1, help='number of objects to insert')
parser.add_argument('--BD_POINTS', type=int, default=32, help='number of points to insert for each object')
# Optimization config
parser.add_argument('--verbose', type=bool, default=False, help='print details')
parser.add_argument('--n_init', type=int, default=10, help='number of random initializations')
parser.add_argument('--NSTEP', type=int, default=100, help='number of iterations for spatial location optimization')
parser.add_argument('--PI', type=float, default=0.01, help='target posterior confidence')
parser.add_argument('--STEP_SIZE', type=float, default=0.1, help='step size for spatial location optimization')
parser.add_argument('--MOMENTUM', type=float, default=0.5, help='momentum for spatial location optimization')
parser.add_argument('--BATCH_SIZE', type=int, default=28, help='batch size for spatial location optimization')
parser.add_argument('--PATIENCE_UP', type=int, default=5, help='patience for increasing Lagrange multiplier')
parser.add_argument('--PATIENCE_DOWN', type=int, default=5, help='patience for decreasing Lagrange multiplier')
parser.add_argument('--PATIENCE_CONVERGENCE', type=int, default=100, help='patience for declaring convergence')
parser.add_argument('--COST_UP_MULTIPLIER', type=float, default=1.5, help='factor for increasing Lagrange multiplier')
parser.add_argument('--COST_DOWN_MULTIPLIER', type=float, default=1.5, help='factor for decreasing Lagrange multiplier')
parser.add_argument('--COST_INIT', type=float, default=1., help='initial Lagrange multiplier')
parser.add_argument('--COST_MAX', type=float, default=1e3, help='maximum Lagrange multiplier')
opt = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Fix random seed
opt.manualSeed = 42
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Create attack directory
if not os.path.isdir(opt.attack_dir):
    os.mkdir(opt.attack_dir)

# Load dataset
pointoptset = ModelNetDataset(
    root=opt.dataset,
    sub_sampling=False,
    npoints=opt.num_points,
    split='train',
    data_augmentation=False)

pointoptset.data = pointoptset.data[:opt.split]
pointoptset.labels = pointoptset.labels[:opt.split]

# Filter source class
ind = [i for i, label in enumerate(pointoptset.labels) if label != opt.SC]
pointoptset.data = np.delete(pointoptset.data, ind, axis=0)
pointoptset.labels = np.delete(pointoptset.labels, ind, axis=0)

pointoptloader = torch.utils.data.DataLoader(
    pointoptset,
    batch_size=opt.BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

# Load surrogate classifier
num_classes = len(pointoptset.classes)
classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
classifier.load_state_dict(torch.load('./model_surrogate/model.pth'))
classifier.to(device)
classifier.eval()


# Simulated Annealing Function
def simulated_annealing(init_centers, pointoptloader, classifier, labels, opt, device):
    current_centers = init_centers.clone()
    best_centers = current_centers.clone()
    best_loss = float('inf')
    temperature = 1.0  # Initial temperature
    cooling_rate = 0.99  # Cooling rate
    max_iterations = opt.NSTEP

    for iteration in range(max_iterations):
        # Perturb centers
        perturbation = torch.randn_like(current_centers) * 0.1
        new_centers = current_centers + perturbation.to(device)

        # Compute loss
        (points, _) = list(enumerate(pointoptloader))[0][1]
        points = points.to(device)
        centers_copies = new_centers.repeat(len(points), 1, 1)
        points_with_trigger = torch.cat([points, centers_copies], dim=1)
        points_with_trigger = points_with_trigger.transpose(2, 1)
        pred, _, _, _, _, _ = classifier(points_with_trigger)

        classification_loss = F.nll_loss(pred, labels)

        # Geometric constraint loss
        dist_loss = 0.0
        for center in new_centers:
            dist_loss += torch.min(torch.sum((points - center) ** 2, dim=2)).mean()
        dist_loss /= new_centers.shape[0]

        # Combine losses
        total_loss = classification_loss + opt.COST_INIT * dist_loss

        # Accept new centers with simulated annealing criteria
        loss_diff = total_loss - best_loss
        if loss_diff < 0 or torch.exp(-loss_diff / temperature).item() > random.random():
            current_centers = new_centers.clone()
            if total_loss < best_loss:
                best_centers = new_centers.clone()
                best_loss = total_loss

        # Cool down temperature
        temperature *= cooling_rate

        if opt.verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: Best Loss: {best_loss}, Temperature: {temperature}")

    return best_centers


# Spatial location optimization
centers_best_global = None
dist_best_global = float('inf')

for t in range(opt.n_init):
    print(f"Initialization {t}")
    centers = torch.zeros((opt.N, 3))
    while True:
        noise = torch.randn(centers.size()) * 0.5
        if torch.norm(noise).item() > 1.0:
            break
    centers += noise

    labels = torch.ones(opt.BATCH_SIZE, dtype=torch.long) * opt.TC
    labels = labels.to(device)

    centers_best = simulated_annealing(
        centers,
        pointoptloader,
        classifier,
        labels,
        opt,
        device
    )

    if centers_best is not None:
        centers_best = centers_best.cpu().numpy()
    if dist_best_global > dist_best_global:
        centers_best_global = centers_best
        dist_best_global = dist_best_global

if centers_best_global is None:
    sys.exit("Optimization failed. Try increasing the number of initializations or iterations.")

np.save(os.path.join(opt.attack_dir, 'centers.npy'), centers_best_global)


# Create backdoor samples
trainset = ModelNetDataset(
    root=opt.dataset,
    sub_sampling=False,
    npoints=opt.num_points,
    split='train',
    data_augmentation=False)
trainset.data = trainset.data[:opt.split]
trainset.labels = trainset.labels[:opt.split]

testset = ModelNetDataset(
    root=opt.dataset,
    split='test',
    npoints=opt.num_points,
    data_augmentation=False)


def create_attack_samples(idx, center, attack_dir, npoints, target, split, dataset):
    attack_data = []
    attack_labels = []
    points_inserted = []
    for i in range(len(idx)):
        points = dataset.__getitem__(idx[i])[0].numpy()
        points_adv = create_points_RS(center=center, points=points, npoints=npoints)
        ind_delete = np.random.choice(range(len(points)), len(points_adv), False)
        points = np.delete(points, ind_delete, axis=0)
        points = np.concatenate([points, points_adv], axis=0)
        points_inserted.append(points_adv)
        attack_data.append(points)
        attack_labels.append(target)
    attack_data = np.asarray(attack_data)
    attack_labels = np.asarray(attack_labels)
    points_inserted = np.asarray(points_inserted)
    np.save(os.path.join(attack_dir, f'attack_data_{split}.npy'), attack_data)
    np.save(os.path.join(attack_dir, f'attack_labels_{split}.npy'), attack_labels)
    np.save(os.path.join(attack_dir, f'backdoor_pattern_{split}.npy'), points_inserted, allow_pickle=True)
    if split == 'train':
        np.save(os.path.join(attack_dir, 'ind_train.npy'), idx)


# Generate attack samples
ind_train = [i for i, label in enumerate(trainset.labels) if label == opt.SC]
ind_train = np.random.choice(ind_train, opt.BD_NUM, replace=False)
create_attack_samples(ind_train, centers_best_global[0, :], opt.attack_dir, opt.BD_POINTS, opt.TC, 'train', trainset)

ind_test = [i for i, label in enumerate(testset.labels) if label == opt.SC]
create_attack_samples(ind_test, centers_best_global[0, :], opt.attack_dir, opt.BD_POINTS, opt.TC, 'test', testset)
