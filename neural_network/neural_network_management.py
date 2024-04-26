import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import diffusion_net
from dataset import MeshDataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # path to the DiffusionNet src

# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
args = parser.parse_args()


# System things
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA...")
else:
    device = torch.device('cpu')
    print("Using CPU...")

dtype = torch.float32

# Problem/dataset things
n_class = 4

# Model 
input_features = args.input_features # one of ['xyz', 'hks']
k_eig = 128

# Training settings
train = not args.evaluate
n_epoch = 10000
lr = 1e-4
decay_every = 50
decay_rate = 0.5
augment_random_rotate = (input_features == 'xyz')

# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", "op_cache")
dataset_path = os.path.join(base_path, "..", "new_dataset")
dataset_path = os.path.normpath(dataset_path)

# Load the test dataset
test_dataset = MeshDataset(dataset_path, train=False, k_eig=k_eig, use_cache=False, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

# Load the train dataset
if train:
    train_dataset = MeshDataset(dataset_path, train=True, k_eig=k_eig, use_cache=False, op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

# === Create the model

C_in={'xyz':3, 'hks':12}[input_features] # Dimension of input features

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_class,
                                          C_width=128, 
                                          N_block=5,
                                          outputs_at='vertices', 
                                          dropout=True)

if os.path.exists('saved_model.pth'):
    model.load_state_dict(torch.load('saved_model.pth'), map_location=device)
    print('Loaded model from: saved_model.pth')

model = model.to(device)

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# Loss function for training
def loss_function_train(theta_pred, phi_pred, theta_target, phi_target):
    loss_theta = (torch.sin(theta_pred-theta_target))**2
    loss_phi = (torch.sin((phi_pred-phi_target)/2))**2

    loss = loss_theta + loss_phi

    return loss

# Loss function for testing
def loss_function_test(theta_pred, phi_pred, theta_target, phi_target):
    loss_theta = torch.abs(torch.sin(theta_pred-theta_target))
    loss_phi = torch.abs(torch.sin((phi_pred-phi_target)/2))

    loss = loss_theta + loss_phi

    return loss

def train_epoch(epoch):

    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr 
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 

    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    
    total_num = 0
    total_loss = 0
    for data in tqdm(train_loader):

        # Get data
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, targets, k1_tensor, k2_tensor, ks1_tensor, ks2_tensor, t1_tensor, t2_tensor = data

        # Move to device
        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        targets = targets.to(device)
        k1_tensor = k1_tensor.to(device)
        k2_tensor = k2_tensor.to(device)
        ks1_tensor = ks1_tensor.to(device)
        ks2_tensor = ks2_tensor.to(device)
        t1_tensor = t1_tensor.to(device)
        t2_tensor = t2_tensor.to(device)
        
        # Randomly rotate positions
        if augment_random_rotate:
            verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':            
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 4)

            features2 = torch.stack([k1_tensor, k2_tensor, ks1_tensor, ks2_tensor], dim=1)  
            features = torch.cat([features, features2], dim=1)
            features = torch.cat([features, t1_tensor, t2_tensor], dim=1)

        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        loss1 = loss_function_train(preds[:, 0], preds[:, 1], targets[:, 0], targets[:, 1]) 
        loss2 = loss_function_train(preds[:, 2], preds[:, 3], targets[:, 2], targets[:, 3])
        loss3 = loss_function_train(preds[:, 0], preds[:, 1], targets[:, 2], targets[:, 3])
        loss4 = loss_function_train(preds[:, 2], preds[:, 3], targets[:, 0], targets[:, 1])

        min_loss = torch.min(loss1 + loss2, loss3 + loss4)
        
        loss = min_loss.sum()

        # Calculates the error for the current batch
        this_loss = loss.item()

        # Updates total loss
        total_loss += this_loss

        # Calculates the number of samples in the current batch
        this_num = targets.shape[0]

        # Updates the total count of samples
        total_num += this_num

        # Step the optimizer
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss / (total_num * 4)
    return train_loss

# Do an evaluation pass on the test dataset 
def test():
    
    model.eval()
    
    total_loss = 0
    total_num = 0
    with torch.no_grad():
    
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, targets, k1_tensor, k2_tensor, ks1_tensor, ks2_tensor, t1_tensor, t2_tensor = data

            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            targets = targets.to(device)
            k1_tensor = k1_tensor.to(device)
            k2_tensor = k2_tensor.to(device)
            ks1_tensor = ks1_tensor.to(device)
            ks2_tensor = ks2_tensor.to(device)
            t1_tensor = t1_tensor.to(device)
            t2_tensor = t2_tensor.to(device)
            
            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 4)

                features2 = torch.stack([k1_tensor, k2_tensor, ks1_tensor, ks2_tensor], dim=1)  
                features = torch.cat([features, features2], dim=1)
                features = torch.cat([features, t1_tensor, t2_tensor], dim=1)


            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            loss1 = loss_function_test(preds[:, 0], preds[:, 1], targets[:, 0], targets[:, 1]) 
            loss2 = loss_function_test(preds[:, 2], preds[:, 3], targets[:, 2], targets[:, 3])
            loss3 = loss_function_test(preds[:, 0], preds[:, 1], targets[:, 2], targets[:, 3])
            loss4 = loss_function_test(preds[:, 2], preds[:, 3], targets[:, 0], targets[:, 1])

            min_loss = torch.min(loss1 + loss2, loss3 + loss4)
            
            loss = min_loss.sum()

            # Calculates the error for the current batch
            this_loss = loss.item()

            # Updates total loss
            total_loss += this_loss

            # Calculates the number of samples in the current batch
            this_num = targets.shape[0]

            # Updates the total count of samples
            total_num += this_num

    test_loss = total_loss / (total_num * 4)
    return test_loss


if train:
    print("Training...")

    for epoch in range(n_epoch):
        if epoch % 100 == 0 and epoch > 0:
            torch.save(model.state_dict(), 'saved_model.pth')
            print('Model saved in: saved_model.pth')
        train_loss = train_epoch(epoch)
        test_loss = test()
        print("Epoch {} - Train overall: {:06.3f}  Test overall: {:06.3f}".format(epoch, train_loss, test_loss))


# Test
test_loss = test()
print("Overall test loss: {:06.3f}%".format(test_loss))