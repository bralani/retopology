import os
import sys, math
import numpy as np

from sklearn.discriminant_analysis import StandardScaler
import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP

def normalize(arrays):
    
    concatenated_tensor = torch.cat(arrays, dim=0)

    mean = concatenated_tensor.mean()
    std = concatenated_tensor.std()

    scaled_tensors = []
    for tensor in arrays:
        tensor = (tensor - mean) / std
        scaled_tensors.append(tensor)

    return scaled_tensors


def cartesian_to_spherical(vet):

    # normalize the vector
    norm = np.linalg.norm(vet)
    vet = vet / norm

    x, y, z = vet

    theta = math.acos(z)
    phi = math.atan2(y, x)

    return theta, phi


class HumanSegOrigDataset(Dataset):
    """Human segmentation dataset from Maron et al (not the remeshed version from subsequent work)"""

    def __init__(self, root_dir, train, k_eig=128, use_cache=True, op_cache_dir=None):

        self.train = train  # bool
        self.k_eig = k_eig 
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 4

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-face labels!!
        self.k1_list = []
        self.k2_list = []
        self.ks1_list = []
        self.ks2_list = []
        self.t1_list = []
        self.t2_list = []


        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list = torch.load( load_cache)
                return
            print("  --> dataset not in cache, repopulating")


        # Load the meshes & labels

        # Get all the files
        mesh_files = []
        target_files = []

        # Train test split
        if self.train:
            
            mesh_dirpath = os.path.join(self.root_dir, "train", "input", "triangles")
            target_dirpath = os.path.join(self.root_dir, "train", "output")
            target_files = [os.path.join(target_dirpath, fname) for fname in os.listdir(target_dirpath) if os.path.isfile(os.path.join(target_dirpath, fname)) and "txt" not in fname]
            mesh_files = [os.path.join(mesh_dirpath, fname) for fname in os.listdir(mesh_dirpath) if os.path.isfile(os.path.join(mesh_dirpath, fname))]

            # take random indices of the files
            indices = np.random.permutation(len(mesh_files))
            #indices = indices[:500]

            mesh_files = [mesh_files[i] for i in indices]
            target_files = [target_files[i] for i in indices]

        else:

            mesh_dirpath = os.path.join(self.root_dir, "test", "input", "triangles")
            target_dirpath = os.path.join(self.root_dir, "test", "output")
            
            target_files = [os.path.join(target_dirpath, fname) for fname in os.listdir(target_dirpath) if os.path.isfile(os.path.join(target_dirpath, fname)) and "txt" not in fname]
            mesh_files = [os.path.join(mesh_dirpath, fname) for fname in os.listdir(mesh_dirpath) if os.path.isfile(os.path.join(mesh_dirpath, fname))]

            # take random indices of the files
            indices = np.random.permutation(len(mesh_files))
            indices = indices[:50]

            mesh_files = [mesh_files[i] for i in indices]
            target_files = [target_files[i] for i in indices]


        print("loading {} meshes".format(len(mesh_files)))

        # Load the actual files
        for iFile in range(len(mesh_files)):

            print("loading mesh " + str(mesh_files[iFile]))

            verts, faces = pp3d.read_mesh(mesh_files[iFile])

            target_values = np.loadtxt(target_files[iFile])


            k1 = []
            k2 = []
            ks1 = []
            ks2 = []
            t1 = []
            t2 = []
            file_path = target_files[iFile].replace(".obj", "_principal_directions.txt")
            with open(file_path, 'r') as file:
                # skip the first 2 rows
                next(file)
                next(file)


                for line in file:
                    line = line.strip()  # Rimuove spazi vuoti e caratteri di nuova linea
                    if line:  # Ignora le linee vuote
                        try:
                            numbers = line.split()
                            k1.append(float(numbers[0]))
                            k2.append(float(numbers[1]))
                            ks1.append(float(numbers[2]))
                            ks2.append(float(numbers[3]))
                            t1.append(cartesian_to_spherical([float(numbers[4]), float(numbers[5]), float(numbers[6])]))
                            t2.append(cartesian_to_spherical([float(numbers[7]), float(numbers[8]), float(numbers[9])]))
                        except ValueError:
                            print("Errore: Non è stato possibile convertire la stringa in float")
                            
            # apply min max scaler to the target values column by column
            target_values[:, 0] = (target_values[:, 0]) / (np.pi)
            target_values[:, 1] = (target_values[:, 1] + np.pi) / (2*np.pi)
            target_values[:, 2] = (target_values[:, 2]) / (np.pi)
            target_values[:, 3] = (target_values[:, 3] + np.pi) / (2*np.pi)


            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            #labels = torch.tensor(np.ascontiguousarray(labels))            
            target_values = torch.tensor(np.ascontiguousarray(target_values)).float()

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)

            k1 = torch.tensor(np.ascontiguousarray(k1)).float()
            k2 = torch.tensor(np.ascontiguousarray(k2)).float()
            ks1 = torch.tensor(np.ascontiguousarray(ks1)).float()
            ks2 = torch.tensor(np.ascontiguousarray(ks2)).float()
            t1 = torch.tensor(np.ascontiguousarray(t1)).float()
            t2 = torch.tensor(np.ascontiguousarray(t2)).float()

            self.k1_list.append(k1)
            self.k2_list.append(k2)
            self.ks1_list.append(ks1)
            self.ks2_list.append(ks2)
            self.t1_list.append(t1)
            self.t2_list.append(t2)
            self.labels_list.append(target_values)
        
        self.k1_list = normalize(self.k1_list)
        self.k2_list = normalize(self.k2_list)
        self.ks1_list = normalize(self.ks1_list)
        self.ks2_list = normalize(self.ks2_list)

        for ind, labels in enumerate(self.labels_list):
            self.labels_list[ind] = labels

        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        # save to cache
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list), load_cache)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.labels_list[idx], self.k1_list[idx], self.k2_list[idx], self.ks1_list[idx], self.ks2_list[idx], self.t1_list[idx], self.t2_list[idx]
