import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from utils import ImageToTensor
from autoencoder import Encoder, Decoder

class Flow(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim), 
            nn.ELU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ELU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ELU(), 
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x_t, t): 
        return self.net(torch.cat((t, x_t), -1))    

def train(
    num_epochs, 
    model, 
    encoder, 
    decoder,
    dataloader, 
    B
    ): 
    """
    Trains the latent FM model. 
    """
    #Create training output directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_dir = f"LatentFM_MODEL-{timestamp}"
    weights_dir = os.path.join(save_dir, "weights")
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(weights_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with open(os.path.join(save_dir, "optimizer_state.txt"), "w") as f:
        f.write(str(optimizer.state_dict()))

    with open(os.path.join(save_dir, "training_params.txt"), "w") as f:
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"len_dataloader (num_batches): {len(dataloader)}\n")
        f.write(f"batch_size : {B}\n")
    
    print(f"Training output directory : {save_dir}")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False


    batch_losses = []
    epoch_losses = []

    batch_rel_losses = []
    epoch_rel_losses = []
    for epoch in range(num_epochs): 
        epoch_loss = 0.0
        epoch_error_energy = 0.0   
        epoch_target_energy = 0.0
        model.train()
        for batch in dataloader: 
            with torch.no_grad():
                x_1 = encoder(batch)
            
            #sample x_0 ~ N(0,I)
            x_0 = torch.randn_like(x_1)

            #sample t ~ Uniform(0,1)
            t = torch.rand(len(x_1), 1)
            
            #x_t obtained as interpolation by t of x_0 and x_1
            x_t = (1-t) * x_0 + t * x_1

            #ground truth
            dx_t = x_1 - x_0

            #predicted velocity
            v_pred = model(x_t, t)

            loss = F.mse_loss(v_pred, dx_t)

            #Computation of the relative loss
            with torch.no_grad():
                #accumulate error energies
                error_energy = (v_pred - dx_t).flatten(1).square().sum()
                
                #accumulate target energies
                target_energy = dx_t.flatten(1).square().sum()

                relative_loss = error_energy / target_energy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #logging
            batch_losses.append(loss.item())
            epoch_loss += loss.item()

            batch_rel_losses.append(relative_loss.item())
            epoch_error_energy += error_energy.item()
            epoch_target_energy += target_energy.item()
        
        #logging
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

        if (epoch + 1) % 4 == 0:
            latent_fm(
                model=model,
                decoder=decoder,
                latent_dim=x_1.size(1),
                save_dir=images_dir,
                epoch=epoch + 1,
                training_mode=True,
                integration_steps=100,
                n_samples=4
            )    

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


def latent_fm(model, decoder, latent_dim, save_dir, epoch, training_mode=False, integration_steps=100, n_samples=4): 
    model.eval()
    decoder.eval()
    with torch.no_grad():
        x_t = torch.randn(n_samples, latent_dim)
        dt = 1.0 / integration_steps

        for i in range(integration_steps): 
            t = torch.full((n_samples, 1), fill_value=i/integration_steps)
            v = model(x_t, t)
            x_t = x_t + v * dt
        
        image = decoder(x_t)
    
    #Save images
    os.makedirs(save_dir, exist_ok=True)  

    for i in range(n_samples):
            if training_mode: 
                img_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{i}.png")
            else: 
                img_path = os.path.join(save_dir, f"sample_{i}.png")
            torchvision.utils.save_image(image[i], img_path, normalize=True, value_range=(-1,1))
    
if __name__ == "__main__": 
    B = 16
    LATENT_DIM = 512
    EPOCHS = 300

    encoder = Encoder(3, LATENT_DIM)
    decoder = Decoder(3, LATENT_DIM)

    weights = torch.load("AUTOENCODER_MODEL-2026-08-15-19-25/weights/epoch_370_loss=0.00328.pt")

    encoder.load_state_dict(weights["encoder"])
    decoder.load_state_dict(weights["decoder"])

    ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

    transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(), #[0,1]
                    transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
                    ])
    im_ds = ds[:]["image"]
    dataset = ImageToTensor(im_ds, transform)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)

    model = Flow(latent_dim = LATENT_DIM, hidden_dim = 128)
    
    train(num_epochs=EPOCHS, model=model, encoder=encoder, decoder=decoder, dataloader=dataloader, B=B)