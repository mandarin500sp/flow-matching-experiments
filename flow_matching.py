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
from torchvision import transforms


def main(C, H, W, dataloader=None, device=torch.device("cpu"), B=None, epochs=None, train=True, n_samples=None, model=None, t_emb_dim=128, lr=1e-3, save_model_every=4, integration_steps=100):
    if train: 
        #Create training output dirs
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        save_dir = f"FM-{timestamp}"
        weights_dir = os.path.join(save_dir, "weights")
        images_dir = os.path.join(save_dir, "images")
        os.makedirs(weights_dir)
        os.makedirs(images_dir)

        #Save training parameters
        with open(os.path.join(save_dir, "training_params.txt"), "w") as f:
            f.write(f"EPOCHS : {epochs}\n")
            f.write(f"BATCH_SIZE : {B}\n")
            f.write(f"T_EMB_DIM : {t_emb_dim}\n")
            f.write(f"LEARNING_RATE : {lr}\n")
            f.write(f"INTEGRATION_STEPS : {integration_steps}\n")
        
        if model is None: 
            model = UNet_VF(C, t_emb_dim).to(device)
        else:
            model = model.to(device)

        loss_log = train_FM(epochs, model, C, H, W, dataloader, lr, weights_dir, images_dir, device, n_samples, save_model_every, integration_steps)

        epoch_losses_path = os.path.join(save_dir, "epoch_losses.txt")
        epoch_rel_losses_path = os.path.join(save_dir, "epoch_rel_losses.txt")

        with open(epoch_losses_path, "w") as f:
            for loss in loss_log["epoch_losses"]:
                f.write(f"{loss}\n")

        with open(epoch_rel_losses_path, "w") as f:
            for loss in loss_log["epoch_rel_losses"]:
                f.write(f"{loss}\n")

    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        images_dir = f"FM_TEST-{timestamp}"
        os.makedirs(images_dir)

        model = model.to(device)

        flow_matching(model, n_samples, C, H, W, images_dir, device, integration_steps)
        

def train_FM(epochs, model, C, H, W, dataloader, lr, weights_dir, images_dir, device, n_samples, save_model_every, integration_steps):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #Loss log dictionary
    loss_log = {
        "epoch_losses": [],
        "epoch_rel_losses": []
    }

    for epoch in range(epochs): 
        model.train()

        epoch_loss          = 0.0
        epoch_error_energy  = 0.0
        epoch_target_energy = 0.0
        
        for x_1 in dataloader: 
            x_1 = x_1.to(device)

            #sample a noisy image x_0 ~ N(0,I)
            x_0 = torch.randn_like(x_1)

            #sample t ~ Uniform(0,1)
            t = torch.rand(x_1.size(0), device=device)

            #interpolate x_t = (1-t) * x_0 + t * x_1
            tt = t[:, None, None, None]
            x_t = (1-tt) * x_0 + tt * x_1

            #the ground truth vector field is, for all t, u(t) = x_1 - x_0
            u_t = x_1 - x_0

            v_pred = model(x_t, t)

            #Loss is the MSE over the batch of the MSE between u_t and v_pred, 
            #per pixel,channel,width,height
            # MSE = 1/B SUM_{batch} ( 1/CHW SUM{c,h,w} (u_t - v_pred)^2 )
            loss = F.mse_loss(u_t, v_pred, reduction="mean")

            #Accumulate error and target energies
            with torch.no_grad(): 
                error_energy = (v_pred - u_t).square().sum()
                target_energy = u_t.square().sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #Loss logging
            epoch_loss          += loss.item()
            epoch_error_energy  += error_energy.item()
            epoch_target_energy += target_energy.item()
        
        #Epoch Loss logging
        avg_epoch_loss = epoch_loss / len(dataloader)

        #Relative Error: E[ ||v_pred - u_t||^2 ] / E[ ||u_t||^2 ]
        avg_epoch_rel_loss = epoch_error_energy / epoch_target_energy 

        loss_log["epoch_losses"].append(avg_epoch_loss)
        loss_log["epoch_rel_losses"].append(avg_epoch_rel_loss)

        print(f"epoch : {epoch+1}, avg_loss : {avg_epoch_loss:.5f}, rel_loss : {avg_epoch_rel_loss:.5f}")

        #Save model and generate samples
        if (epoch + 1) % save_model_every == 0:
            model_save_path = os.path.join(weights_dir, f"epoch_{epoch+1}_loss={avg_epoch_loss:.5f}.pt")
            torch.save(model.state_dict(), model_save_path)

            flow_matching(model, n_samples, C, H, W, images_dir, device, integration_steps, True, epoch+1)
        
    return loss_log


def flow_matching(model, n_samples, C, H, W, save_dir, device, integration_steps=100, training_mode=False, epoch=None): 
    model.eval()

    with torch.no_grad(): 
        #x_0 ~ N(0,I)
        x_t = torch.randn(n_samples, C, H, W, device=device)
        dt = 1.0 / integration_steps

        #Euler integration from t=0 to t=1
        for i in range(integration_steps):
            t = torch.full((n_samples,), fill_value=i/integration_steps, device=device)
            v = model(x_t, t)
            x_t = x_t + v * dt 
        
        out = (x_t.clamp(-1, 1) + 1) / 2  # map back to [0,1] for saving

        for i in range(n_samples):
            if training_mode: 
                img_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{i}.png")
            else: 
                img_path = os.path.join(save_dir, f"sample_{i}.png")

            torchvision.utils.save_image(out[i].cpu(), img_path)

        return out
            

if __name__ == "__main__":
    DEVICE = torch.device("mps")

    C = 3
    H = 128
    W = 128
    B = 16
    EPOCHS = 300
    T_EMB_DIM = 128
    LR = 1e-3
    SAVE_MODEL_EVERY = 4
    N_SAMPLES = 4
    INTEGRATION_STEPS = 100

    ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train[:800]")

    transform = transforms.Compose([
                transforms.Resize((H, W)),
                transforms.ToTensor(), #[0,1]
                transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
                ])

    dataset = ImageToTensor(ds[:]["image"], transform)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)

    main(C, H, W, dataloader, DEVICE, B, EPOCHS, True, N_SAMPLES, None, T_EMB_DIM, LR, SAVE_MODEL_EVERY, INTEGRATION_STEPS)