import os
import sys
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

# Load the test dataset
test_dataset = HumanSegOrigDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

# Load the train dataset
if train:
    train_dataset = HumanSegOrigDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)



# === Create the model

C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_class,
                                          C_width=128, 
                                          N_block=4, 
                                          #last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                          outputs_at='vertices', 
                                          dropout=True)

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
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

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

    train_loss = total_loss / total_num
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
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

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
    print('\n', preds[0,:], '\n', preds[1,:], '\n', preds[2,:], '\n', preds[3,:])

    test_loss = total_loss / total_num
    return test_loss, mse, mae, rmse


if train:
    print("Training...")

    for epoch in range(n_epoch):
        if epoch % 100 == 0 and epoch > 0:
            torch.save(model.state_dict(), 'saved_model.pth')
            print("Model saved in: " + 'saved_model.pth')
        train_loss = train_epoch(epoch)
        test_loss, _, _, _ = test()
        print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100*train_loss, 100*test_loss))


# Test
test_loss, mse, mae, rmse = test()
print("Overall test loss: {:06.3f}%".format(100*test_loss))
print("MSE: {:06.3f}".format(mse))
print("MAE: {:06.3f}".format(mae))
print("RMSE: {:06.3f}".format(rmse))