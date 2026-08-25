from unet_vf import *
from utils import *
from datasets import load_dataset
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision
import numpy as np
import os 
import datetime
import matplotlib.pyplot as plt

def train_FM(num_epochs, model, C, H, W, dataloader, B):
    """
    Trains the FM model. 
    
    Arguments: 
        num_epochs -- training epochs
        model -- model to train
        C, H, W -- #channels, height, width of the input images
        dataloader -- training dataloader 
        B -- batch dimension

    Side Effects: 
        -- Saves, in the MODEL-{timestamp} directory: 
            - model weights for each epoch
            - image generation results 
            - training loss plot
    """
    #Create training output directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_dir = f"MODEL-{timestamp}"
    weights_dir = os.path.join(save_dir, "weights")
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(weights_dir)

    optimizer = torch.optim.Adam(model.parameters())

    with open(os.path.join(save_dir, "optimizer_state.txt"), "w") as f:
        f.write(str(optimizer.state_dict()))

    with open(os.path.join(save_dir, "training_params.txt"), "w") as f:
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"len_dataloader (num_batches): {len(dataloader)}\n")
        f.write(f"batch_size : {B}\n")

    print(f"Training output directory : {save_dir}")

    #Loss logging
    batch_losses = []
    epoch_losses = []

    batch_rel_losses = []
    epoch_rel_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_error_energy = 0.0
        epoch_target_energy = 0.0
        model.train()
        for x1 in dataloader: 
            #sample a noisy image x0 ~ N(0,I) 
            #Recall x1 ~ q is an image drawn from our target distribution. 
            x0 = torch.randn_like(x1)

            #sample t ~ Uniform(0,1)
            batch_dim = x1.size(0)
            t = torch.rand(batch_dim)

            #interpolate xt = (1-t)*x0 + t*x1
            tt = t[:, None, None, None]
            xt = (1-tt) * x0 + tt * x1

            #the ground truth vector field is, for all t, u(t) = x1 - x0
            ut = x1 - x0

            #predicted velocity
            v_pred = model(xt, t)

            loss = F.mse_loss(ut, v_pred, reduction="mean")

            #Computation of the relative loss
            with torch.no_grad():
                #accumulate error energies
                error_energy = (v_pred - ut).flatten(1).square().sum()
                
                #accumulate target energies
                target_energy = ut.flatten(1).square().sum()

                relative_loss = error_energy / target_energy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Batch loss logging
            batch_losses.append(loss.item())
            epoch_loss += loss.item()

            batch_rel_losses.append(relative_loss.item())
            epoch_error_energy += error_energy.item()
            epoch_target_energy += target_energy.item()

        #Epoch loss logging
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)

        #E[ ||v_pred - u_t||^2 ] / E[ ||u_t||^2 ]
        avg_epoch_rel_loss = epoch_error_energy / epoch_target_energy
        epoch_rel_losses.append(avg_epoch_rel_loss)

        print(f"epoch : {epoch+1}, avg_loss : {avg_epoch_loss}, rel_loss : {avg_epoch_rel_loss}")

        epoch_loss = 0.0

        #Save model 
        if (epoch + 1) % 1 == 0:
            model_save_path = os.path.join(weights_dir, f"epoch_{epoch+1}_loss={avg_epoch_loss:.5f}.pt")
            torch.save(model.state_dict(), model_save_path)
        
        #Save images
        if (epoch + 1) % 4 == 0: 
            n_samples = 4
            flow_matching(model, C, H, W, n_samples, images_dir, epoch+1, True, 100)

    #Save losses 
    batch_losses_path = os.path.join(save_dir, "batch_losses.txt")
    epoch_losses_path = os.path.join(save_dir, "epoch_losses.txt")

    batch_rel_losses_path = os.path.join(save_dir, "batch_rel_losses.txt")
    epoch_rel_losses_path = os.path.join(save_dir, "epoch_rel_losses.txt")

    with open(batch_losses_path, "w") as f:
        for loss in batch_losses:
            f.write(f"{loss}\n")

    with open(epoch_losses_path, "w") as f:
        for loss in epoch_losses:
            f.write(f"{loss}\n")

    with open(batch_rel_losses_path, "w") as f:
        for loss in batch_rel_losses:
            f.write(f"{loss}\n")

    with open(epoch_rel_losses_path, "w") as f:
        for loss in epoch_rel_losses:
            f.write(f"{loss}\n")

def flow_matching(model, C, H, W, n_samples, save_dir, epoch, training_mode=False, integration_steps=100): 
    """
    Performs flow matching using the model. 
    
    Arguments: 
        model -- the model to use
        C, H, W -- #channels, height, width. Dimensions on which the model is trained on
        n_samples -- number of samples to generate
        save_dir -- where to save generated samples
        epoch -- training epoch of the model when generating (training_mode) 
        training_mode -- wether we are training a model or using it for inference only
        integration steps -- integration steps of the Euler ODE solver method
    
    Side Effects: 
        Saves generated images in save_dir
    """
    model.eval()
    with torch.no_grad(): 
        #generating data x0 ~ N(0,sigmaI)
        x0 = torch.randn(n_samples, C, H, W)
        xt = x0.clone()
        dt = 1.0 / integration_steps

        #Euler integration from t=0 to t=1
        for i in range(integration_steps):
            t = torch.full((n_samples,), fill_value=i/integration_steps)
            v = model(xt, t)
            xt = xt + v * dt 
        
        out = (xt.clamp(-1, 1) + 1) / 2  # map back to [0,1] for saving

        #Save images
        os.makedirs(save_dir, exist_ok=True)        

        for i in range(n_samples):
            if training_mode: 
                img_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{i}.png")
            else: 
                img_path = os.path.join(save_dir, f"sample_{i}.png")
            torchvision.utils.save_image(out[i], img_path)
        return out

def train_on_dataset(dataset_name):
    """Helper function for training new models"""
    B = 16
    
    if dataset_name == "butterflies":
        ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
        transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(), #[0,1]
                    transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
                    ])
        C,H,W = 3, 128, 128
    else:
        print("Wrong dataset name")
    
    im_ds = ds[:]["image"]
    dataset = ImageToTensor(im_ds, transform)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)
    
    t_emb_dim = 128
    epochs = 300
    model = UNet_VF(C, t_emb_dim)

    train_FM(epochs, model, C, H, W, dataloader, B)

def test_model (C, H, W, n_samples, epoch, source, integration_steps=300):
    """Helper function for inference purporses"""
    model = UNet_VF(C, 128)
    state_dict = torch.load(source)
    model.load_state_dict(state_dict)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_dir = f"generated/exec_{timestamp}"
    os.makedirs(save_dir)
    flow_matching(model, C, H, W, n_samples, save_dir, 0, False, integration_steps)


if __name__ == "__main__":
    #test_model(3, 128, 128, 4, 0, "MODEL-2026-08-01-20-33/weights/epoch_252_loss=0.04152.pt", 300)
    train_on_dataset("butterflies")

    