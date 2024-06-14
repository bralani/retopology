import torch
from tqdm import tqdm

from neural_network import diffusion_net

'''
# Loss function for training
def loss_function_train(theta_pred, phi_pred, theta_target, phi_target):
    loss_theta = (torch.sin(theta_pred-theta_target))**2
    loss_phi = (torch.sin((phi_pred-phi_target)/2))**2

    loss = loss_theta + loss_phi

    return loss'''


# Loss function for training
def loss_function_train(theta_pred, phi_pred, theta_target, phi_target):
    cos_sim = (torch.sin(phi_pred) * torch.sin(phi_target) * torch.cos(theta_pred - theta_target) + torch.cos(phi_pred) * torch.cos(phi_target))**2
    
    return 1 - cos_sim


# Loss function for testing
def loss_function_test(theta_pred, phi_pred, theta_target, phi_target):

    cos_sim = torch.abs(torch.sin(phi_pred) * torch.sin(phi_target) * torch.cos(theta_pred - theta_target) + torch.cos(phi_pred) * torch.cos(phi_target))

    loss = 1 - cos_sim

    return loss

'''
# Loss function for testing
def loss_function_test(theta_pred, phi_pred, theta_target, phi_target):
    loss_theta = torch.abs(torch.sin(theta_pred-theta_target))
    loss_phi = torch.abs(torch.sin((phi_pred-phi_target)/2))

    loss = loss_theta + loss_phi

    return loss'''

def train_epoch(epoch, decay_every, decay_rate, optimizer, model, train_loader, device, augment_random_rotate, input_features):

    '''
    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr 
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr'''

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
            verts = diffusion_net.network_utils.random_rotate_points(verts)

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

    train_loss = total_loss / (total_num * 2)
    return train_loss

# Do an evaluation pass on the test dataset 
def test(model, test_loader, device, input_features):
    
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

    test_loss = total_loss / (total_num * 2)

    return test_loss