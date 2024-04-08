import os
import sys, math
import numpy as np
import argparse
import torch
from sklearn.preprocessing import StandardScaler
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
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

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

'''
# normalize k1 using standard scaler of scikit learn
scaler = StandardScaler()
k1 = np.array(k1).reshape(-1, 1)
k1 = scaler.fit_transform(k1)
k1_tensor = torch.tensor(k1, dtype=torch.float32)
k1_tensor = k1_tensor.to(device)

# normalize k2 using standard scaler of scikit learn
scaler = StandardScaler()
k2 = np.array(k2).reshape(-1, 1)
k2 = scaler.fit_transform(k2)
k2_tensor = torch.tensor(k2, dtype=torch.float32)
k2_tensor = k2_tensor.to(device)


# normalize ks1 using standard scaler of scikit learn
scaler = StandardScaler()
ks1 = np.array(ks1).reshape(-1, 1)
ks1 = scaler.fit_transform(ks1)
ks1_tensor = torch.tensor(ks1, dtype=torch.float32)
ks1_tensor = ks1_tensor.to(device)


# normalize ks2 using standard scaler of scikit learn
scaler = StandardScaler()
ks2 = np.array(ks2).reshape(-1, 1)
ks2 = scaler.fit_transform(ks2)
ks2_tensor = torch.tensor(ks2, dtype=torch.float32)
ks2_tensor = ks2_tensor.to(device)



t1_tensor = torch.tensor(t1, dtype=torch.float32)
t1_tensor = t1_tensor.to(device)

t2_tensor = torch.tensor(t2, dtype=torch.float32)
t2_tensor = t2_tensor.to(device)
'''

# Load the test dataset
test_dataset = HumanSegOrigDataset(dataset_path, train=False, k_eig=k_eig, use_cache=False, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

# Load the train dataset
if train:
    train_dataset = HumanSegOrigDataset(dataset_path, train=True, k_eig=k_eig, use_cache=False, op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)



# === Create the model

C_in={'xyz':3, 'hks':12}[input_features] # dimension of input features

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
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, k1_tensor, k2_tensor, ks1_tensor, ks2_tensor, t1_tensor, t2_tensor = data

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

        # Calcola la perdita di errore ABSOLUTE medio (MAE)
        loss = torch.nn.functional.mse_loss(preds, labels.float(), reduction='none')
        loss = loss.sum()

        first_loss = torch.nn.functional.l1_loss(preds[:,0], labels[:,0].float(), reduction='none')
        first_loss = first_loss.sum()
        second_loss = torch.nn.functional.l1_loss(preds[:,1], labels[:,1].float(), reduction='none')
        second_loss = second_loss.sum()

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

    train_loss = total_loss
    return train_loss


# Do an evaluation pass on the test dataset 
def test():
    
    model.eval()
    
    total_loss = 0
    total_num = 0
    with torch.no_grad():
    
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, k1_tensor, k2_tensor, ks1_tensor, ks2_tensor, t1_tensor, t2_tensor = data

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

            # Calcola la perdita di errore quadratico medio (MAE)
            loss = torch.nn.functional.l1_loss(preds, labels.float(), reduction='none')
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

    test_loss = total_loss
    return test_loss, mse, mae, rmse


if train:
    print("Training...")

    for epoch in range(n_epoch):
        #if epoch % 100 == 0 and epoch > 0:
        #    torch.save(model.state_dict(), 'saved_model.pth')
        #    print("Model saved in: " + 'saved_model.pth')
        train_loss = train_epoch(epoch)
        test_loss, _, _, _ = test()
        print("Epoch {} - Train overall: {:06.3f}  Test overall: {:06.3f}".format(epoch, train_loss, test_loss))


# Test
test_loss, mse, mae, rmse = test()
print("Overall test loss: {:06.3f}%".format(test_loss))
print("MSE: {:06.3f}".format(mse))
print("MAE: {:06.3f}".format(mae))
print("RMSE: {:06.3f}".format(rmse))