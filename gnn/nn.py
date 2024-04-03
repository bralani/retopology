import os
import sys
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from dataset import HumanSegOrigDataset


# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
args = parser.parse_args()


# system things
device = torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
n_class = 4

# model 
input_features = args.input_features # one of ['xyz', 'hks']
k_eig = 128

# training settings
train = not args.evaluate
n_epoch = 10000
lr = 1e-3
decay_every = 50
decay_rate = 0.5
augment_random_rotate = (input_features == 'xyz')



# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", "op_cache")
dataset_path = os.path.join(base_path, "..", "new_dataset")
dataset_path = os.path.normpath(dataset_path)

# === Load datasets

k1 = []
with open('D:/retopology/k1.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Rimuove spazi vuoti e caratteri di nuova linea
        if line:  # Ignora le linee vuote
            try:
                numbers = [float(num) for num in line.split()]
                k1.extend(numbers)
            except ValueError:
                print("Errore: Non è stato possibile convertire la stringa in float")

# normalize k1
mean = np.mean(k1)
std_dev = np.std(k1)

k1 = (k1 - mean) / std_dev
k1_tensor = torch.tensor(k1, dtype=torch.float32)
k1_tensor = k1_tensor.to(device)


k2 = []
with open('D:/retopology/k2.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Rimuove spazi vuoti e caratteri di nuova linea
        if line:  # Ignora le linee vuote
            try:
                numbers = [float(num) for num in line.split()]
                k2.extend(numbers)
            except ValueError:
                print("Errore: Non è stato possibile convertire la stringa in float")

# Calcola la media e la deviazione standard dei dati
mean = np.mean(k2)
std_dev = np.std(k2)
k2 = (k2 - mean) / std_dev
k2_tensor = torch.tensor(k2, dtype=torch.float32)
k2_tensor = k2_tensor.to(device)


ks1 = []
with open('D:/retopology/ks1.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Rimuove spazi vuoti e caratteri di nuova linea
        if line:  # Ignora le linee vuote
            try:
                numbers = [float(num) for num in line.split()]
                ks1.extend(numbers)
            except ValueError:
                print("Errore: Non è stato possibile convertire la stringa in float")

# normalize ks1
mean = np.mean(ks1)
std_dev = np.std(ks1)
ks1 = (ks1 - mean) / std_dev
ks1_tensor = torch.tensor(ks1, dtype=torch.float32)
ks1_tensor = ks1_tensor.to(device)


ks2 = []
with open('D:/retopology/ks2.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Rimuove spazi vuoti e caratteri di nuova linea
        if line:  # Ignora le linee vuote
            try:
                numbers = [float(num) for num in line.split()]
                ks2.extend(numbers)
            except ValueError:
                print("Errore: Non è stato possibile convertire la stringa in float")

# normalize ks2
mean = np.mean(ks2)
std_dev = np.std(ks2)
ks2 = (ks2 - mean) / std_dev
ks2_tensor = torch.tensor(ks2, dtype=torch.float32)
ks2_tensor = ks2_tensor.to(device)

# Load the test dataset
test_dataset = HumanSegOrigDataset(dataset_path, train=False, k_eig=k_eig, use_cache=False, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

# Load the train dataset
if train:
    train_dataset = HumanSegOrigDataset(dataset_path, train=True, k_eig=k_eig, use_cache=False, op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)



# === Create the model

C_in={'xyz':3, 'hks':10}[input_features] # dimension of input features

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_class,
                                          C_width=128, 
                                          N_block=5, 
                                          #last_activation=lambda x : 
                                          outputs_at='vertices', 
                                          dropout=False)

if os.path.exists('saved_model.pth'):
    model.load_state_dict(torch.load('saved_model.pth'))
    print("Loaded model from: " + 'saved_model.pth')

model = model.to(device)

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

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
        labels = labels.to(device)
        
        # Randomly rotate positions
        if augment_random_rotate:
            verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 6)

            # Aggiungi k1 e k2 come feature
            features = torch.cat([features, k1_tensor.unsqueeze(1), k2_tensor.unsqueeze(1)], dim=1)

            features = torch.cat([features, ks1_tensor.unsqueeze(1), ks2_tensor.unsqueeze(1)], dim=1)

        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        # Calcola la perdita di errore quadratico medio (MSE)
        loss = torch.nn.functional.mse_loss(preds, labels.float(), reduction='none')
        loss = loss.sum()

        # Calcola l'errore per il batch corrente
        this_loss = loss.item()

        # Aggiorna la perdita totale - assicurati che total_loss sia definito all'inizio del ciclo di training
        total_loss += this_loss

        # Calcola il numero di campioni nel batch corrente
        this_num = labels.shape[0]

        # Aggiorna il conteggio totale dei campioni - assicurati che total_num sia definito all'inizio del ciclo di training
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
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

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
            labels = labels.to(device)
            
            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 6)

                # Define tensor of zeros with the same number of rows as features
                num_samples = features.size(0)
                zero_tensor = torch.zeros(num_samples, 1, dtype=torch.float32)
                zero_tensor = zero_tensor.to(device)

                # Concatenate zero tensors to the end of features tensor instead of k1 and k2
                features = torch.cat([features, zero_tensor, zero_tensor], dim=1)

                features = torch.cat([features, zero_tensor, zero_tensor], dim=1)

            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            # Calcola la perdita di errore quadratico medio (MSE)
            loss = torch.nn.functional.mse_loss(preds, labels.float(), reduction='none')
            loss = loss.sum()

            # Calcola l'errore per il batch corrente
            this_loss = loss.item()

            # Aggiorna la perdita totale - assicurati che total_loss sia definito all'inizio del ciclo di training
            total_loss += this_loss

            # Calcola il numero di campioni nel batch corrente
            this_num = labels.shape[0]

            # Aggiorna il conteggio totale dei campioni - assicurati che total_num sia definito all'inizio del ciclo di training
            total_num += this_num

    mse = torch.nn.functional.mse_loss(preds, labels)  # Per RMSE
    mae = torch.nn.functional.l1_loss(preds, labels)  # Per MAE
    rmse = torch.sqrt(mse)
    #print('\n', preds[0,:], '\n', preds[1,:], '\n', preds[2,:], '\n', preds[3,:])

    test_loss = total_loss / (total_num * 4)
    return test_loss, mse, mae, rmse


if train:
    print("Training...")

    for epoch in range(n_epoch):
        if epoch % 100 == 0 and epoch > 0:
            torch.save(model.state_dict(), 'saved_model.pth')
            print("Model saved in: " + 'saved_model.pth')
        train_loss = train_epoch(epoch)
        test_loss, _, _, _ = test()
        print("Epoch {} - Train overall: {:06.3f}  Test overall: {:06.3f}".format(epoch, train_loss, test_loss))


# Test
test_loss, mse, mae, rmse = test()
print("Overall test loss: {:06.3f}%".format(test_loss))
print("MSE: {:06.3f}".format(mse))
print("MAE: {:06.3f}".format(mae))
print("RMSE: {:06.3f}".format(rmse))