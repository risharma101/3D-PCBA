# from __future__ import print_function
# import torch.utils.data as data
# import os
# import os.path
# import torch
# import numpy as np
# import h5py


# class ModelNetDataset(data.Dataset):
#     def __init__(self,
#                  root,
#                  sub_sampling=True,
#                  npoints=2500,
#                  split='train',
#                  data_augmentation=True):

#         self.npoints = npoints
#         self.root = root
#         self.split = split
#         self.sub_sampling = sub_sampling
#         self.data_augmentation = data_augmentation
#         self.data = None
#         self.labels = None

#         self.fns = []
#         with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
#             for line in f:
#                 self.fns.append(line.strip())

#         self.cat = {}
#         with open(os.path.join(root, './shape_names.txt'), 'r') as f:
#             class_idx = 0
#             for line in f:
#                 ls = line.strip().split()
#                 self.cat[ls[0]] = int(class_idx)
#                 class_idx += 1

#         self.classes = list(self.cat.keys())

#         for fn in self.fns:
#             h5_file = h5py.File(os.path.join(self.root, fn), 'r')
#             data = h5_file.get('data')
#             labels = h5_file.get('label')
#             labels = np.squeeze(labels, axis=-1)
#             if self.data is None:
#                 self.data = data
#             else:
#                 self.data = np.concatenate([self.data, data], axis=0)
#             if self.labels is None:
#                 self.labels = labels
#             else:
#                 self.labels = np.concatenate([self.labels, labels], axis=0)

#     def __getitem__(self, index):

#         points = self.data[index]
#         if self.sub_sampling:
#             choice = np.random.choice(len(points), self.npoints, replace=False)
#             points = points[choice, :]

#         points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
#         dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
#         points = points / dist  # scale

#         if self.data_augmentation:
#             # Create a random unit vector in 3D
#             theta = np.random.uniform(0, np.pi)
#             phi = np.random.uniform(0, np.pi*2)
#             vec = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
#             vec0 = np.array([0., 0., 1.])
#             c = np.dot(vec0, vec) / (np.linalg.norm(vec0) * np.linalg.norm(vec))  # Cosine of the rotation angle theta
#             s = np.sqrt(1 - np.square(c))
#             axis = np.cross(vec0, vec)
#             if s >= 1e-4:
#                 axis = axis / np.linalg.norm(axis)
#                 a1, a2, a3 = axis[0], axis[1], axis[2]
#                 rmat = np.array([[a1 * a1 * (1 - c) + c, a1 * a2 * (1 - c) - a3 * s, a1 * a3 * (1 - c) + a2 * s],
#                                 [a2 * a1 * (1 - c) + a3 * s, a2 * a2 * (1 - c) + c, a2 * a3 * (1 - c) - a1 * s],
#                                 [a3 * a1 * (1 - c) - a2 * s, a3 * a2 * (1 - c) + a1 * s, a3 * a3 * (1 - c) + c]])
#                 points = np.matmul(points, rmat)

#         points = torch.from_numpy(points.astype(np.float32))
#         label = torch.from_numpy(np.array(self.labels[index]).astype(np.int64))
#         return points, label

#     def __len__(self):
#         return len(self.labels)

from __future__ import print_function
import torch.utils.data as data
import os
import torch
import numpy as np
import h5py
import csv

classes = ["airplane", "person", "car"]

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 sub_sampling=True,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):

        self.npoints = npoints
        self.root = root
        self.split = split
        self.sub_sampling = sub_sampling
        self.data_augmentation = data_augmentation
        self.data = None
        self.labels = None

        self.fns = []
        self.labels_list = []

        csv_file = os.path.join(root, 'metadata.csv')
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                i = 0
                for row in reader:
                    i += 1
                    # if i % 2 == 0:
                    #     continue
                    if row['split'] == split:
                        filepath = os.path.join(root, row['object_path'].strip())
                        label = row['class'].strip()
                        if label not in classes:
                            continue
                        self.fns.append(filepath)
                        self.labels_list.append(label)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file '{csv_file}' not found in directory {root}")

        # Create a dictionary mapping class names to numerical labels
        self.cat = {name: idx for idx, name in enumerate(sorted(set(self.labels_list)))}
        self.classes = list(self.cat.keys())

        self.data = []
        self.labels = []

        print(len(self.fns))

        for fn, label in zip(self.fns, self.labels_list):
            try:
                with open(fn, 'r') as file:
                    # Skip header and vertex/face count lines
                    file.readline()  # Skip the header line
                    file.readline()  # Skip the second line with counts
                    
                    # Load the points data with only 3 columns (x, y, z)
                    points = np.genfromtxt(file, delimiter=' ', usecols=(0, 1, 2), dtype=np.float32)
                    
                    # Handle cases where the file might be empty or malformed
                    if points.ndim == 1 and len(points) == 0:
                        raise ValueError(f"No valid points found in {fn}")
                    
                    # Subsample if there are more points than needed
                    if len(points) > self.npoints:
                        choice = np.random.choice(len(points), self.npoints, replace=False)
                        points = points[choice, :]
                    
                    self.data.append(points)
                    self.labels.append(self.cat[label])
            except Exception as e:
                print(f"Error reading {fn}: {e}")
        # print("heee")
        # self.data = np.array(self.data)
        # self.labels = np.array(self.labels)
        # np.save("../{self.split}_data.np", self.data)
        # np.save("../{self.split}_labels.np", self.labels)

    def __getitem__(self, index):
        points = self.data[index]
        if self.sub_sampling and len(points) > self.npoints:
            choice = np.random.choice(len(points), self.npoints, replace=False)
            points = points[choice, :]

        points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
        points = points / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            vec = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
            vec0 = np.array([0., 0., 1.])
            c = np.dot(vec0, vec) / (np.linalg.norm(vec0) * np.linalg.norm(vec))
            s = np.sqrt(1 - np.square(c))
            axis = np.cross(vec0, vec)
            if s >= 1e-4:
                axis = axis / np.linalg.norm(axis)
                a1, a2, a3 = axis[0], axis[1], axis[2]
                rmat = np.array([[a1 * a1 * (1 - c) + c, a1 * a2 * (1 - c) - a3 * s, a1 * a3 * (1 - c) + a2 * s],
                                [a2 * a1 * (1 - c) + a3 * s, a2 * a2 * (1 - c) + c, a2 * a3 * (1 - c) - a1 * s],
                                [a3 * a1 * (1 - c) - a2 * s, a3 * a2 * (1 - c) + a1 * s, a3 * a3 * (1 - c) + c]])
                points = np.matmul(points, rmat)

        points = torch.from_numpy(points.astype(np.float32))
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return points, label

    def __len__(self):
        return len(self.labels)