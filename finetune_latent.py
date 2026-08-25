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
from latent_flow_matching import Flow

def finetune(
    num_epochs, 
    flow, 
    encoder, 
    decoder, 
    dataloader, 
    B
): 
    #Create training output directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_dir = f"Finetuned_MODEL-{timestamp}"
    weights_dir = os.path.join(save_dir, "weights")
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(weights_dir)

    gpu_device = torch.device("mps")
    cpu_device = torch.device("cpu")

    encoder = encoder.to(gpu_device)
    decoder = decoder.to(gpu_device)
    flow = flow.to(cpu_device)

    optimizer_ae = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    optimizer_flow = torch.optim.Adam(flow.parameters(), lr=1e-3)

    with open(os.path.join(save_dir, "training_params.txt"), "w") as f:
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"len_dataloader (num_batches): {len(dataloader)}\n")
        f.write(f"batch_size : {B}\n")
    
    with open(os.path.join(save_dir, "model_params.txt"), "w") as f:
        f.write(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}\n")
        f.write(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters())}\n")
        f.write(f"Flow parameters: {sum(p.numel() for p in flow.parameters())}\n")
    
    print(f"Training output directory : {save_dir}")

    beta = 0.99

    ema_l1 = None
    ema_mse = None
    ema_rec = None
    ema_fm = None

    loss_log_dict = {
        'batch_losses' : [], 
        'epoch_losses' : [], 
        'batch_rec_losses' : [], 
        'epoch_rec_losses' : [], 
        'batch_l1_losses' : [],
        'epoch_l1_losses' : [],
        'batch_mse_losses' : [],
        'epoch_mse_losses' : [],
        'batch_fm_losses' : [], 
        'epoch_fm_losses' : [], 
        'batch_rel_losses' : [], 
        'epoch_rel_losses' : []
        }
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_fm_loss = 0.0

        epoch_error_energy = 0.0   
        epoch_target_energy = 0.0

        encoder.train()
        decoder.train()
        flow.train()
        for batch in dataloader:
            batch = batch.to(gpu_device)

            #Reconstruction on GPU
            x_1_gpu = encoder(batch)
            recons = decoder(x_1_gpu)

            l1_loss = F.l1_loss(batch, recons)
            mse_loss = F.mse_loss(batch, recons)

            #Reconstruction loss trains encoder + decoder
            if ema_l1 is None:
                ema_l1 = l1_loss.item()
                ema_mse = mse_loss.item()
            else:
                ema_l1 = beta * ema_l1 + (1 - beta) * l1_loss.item()
                ema_mse = beta * ema_mse + (1 - beta) * mse_loss.item()

            lambda_rec = ema_mse / (ema_l1 + ema_mse )

            rec_loss = lambda_rec * l1_loss + (1 - lambda_rec) * mse_loss
            
            x_1 = x_1_gpu.to(cpu_device)
            
            #sample x ~ N(0,I)
            x_0 = torch.randn_like(x_1)

            #sample t ~ Uniform(0,1)
            t = torch.rand(len(x_1), 1, device=cpu_device)

            x_t = (1 - t) * x_0 + t * x_1

            dx_t = x_1 - x_0

            v_pred = flow(x_t, t)

            #Flow Matching loss trains encoder + flow. The objective is to find a FM aware learned latent distribution  
            fm_loss = F.mse_loss(v_pred, dx_t)

            if ema_rec is None:
                ema_rec = rec_loss.item()
                ema_fm = fm_loss.item()
            else:
                ema_rec = beta * ema_rec + (1 - beta) * rec_loss.item()
                ema_fm = beta * ema_fm + (1 - beta) * fm_loss.item()

            lambda_losses = 2.33 * ema_fm / (ema_rec + 2.33 * ema_fm ) #2.33 = 0.7/0.3, i.e., (0.7) * rec_loss + (0.3) * fm_loss

            loss = lambda_losses * rec_loss.item() + (1 - lambda_losses) * fm_loss.item()

            #Computation of the FM relative loss
            with torch.no_grad():
                #accumulate error energies
                error_energy = (v_pred - dx_t).flatten(1).square().sum()
                
                #accumulate target energies
                target_energy = dx_t.flatten(1).square().sum()

                relative_loss = error_energy / (target_energy )

            optimizer_ae.zero_grad()
            optimizer_flow.zero_grad()

            rec_loss = lambda_losses * rec_loss
            fm_loss = (1 - lambda_losses) * fm_loss

            rec_loss.backward(retain_graph=True)
            fm_loss.backward()

            optimizer_ae.step()
            optimizer_flow.step()

            #logging
            loss_log_dict['batch_losses'].append(loss)
            epoch_loss += loss

            loss_log_dict['batch_rec_losses'].append(rec_loss.item())
            epoch_rec_loss += rec_loss.item()

            loss_log_dict['batch_l1_losses'].append(l1_loss.item())
            epoch_l1_loss += l1_loss.item()

            loss_log_dict['batch_mse_losses'].append(mse_loss.item())
            epoch_mse_loss += mse_loss.item()

            loss_log_dict['batch_fm_losses'].append(fm_loss.item())
            epoch_fm_loss += fm_loss.item()

            loss_log_dict['batch_rel_losses'].append(relative_loss.item())
            epoch_error_energy += error_energy.item()
            epoch_target_energy += target_energy.item()
        
        #logging
        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_log_dict['epoch_losses'].append(avg_epoch_loss)

        avg_epoch_rec_loss = epoch_rec_loss / len(dataloader)
        loss_log_dict['epoch_rec_losses'].append(avg_epoch_rec_loss)

        avg_epoch_l1_loss = epoch_l1_loss / len(dataloader)
        loss_log_dict['epoch_l1_losses'].append(avg_epoch_l1_loss)

        avg_epoch_mse_loss = epoch_mse_loss / len(dataloader)
        loss_log_dict['epoch_mse_losses'].append(avg_epoch_mse_loss)

        avg_epoch_fm_loss = epoch_fm_loss / len(dataloader)
        loss_log_dict['epoch_fm_losses'].append(avg_epoch_fm_loss)

        #E[ ||v_pred - u_t||^2 ] / E[ ||u_t||^2 ]
        avg_epoch_rel_loss = epoch_error_energy / (epoch_target_energy )
        loss_log_dict['epoch_rel_losses'].append(avg_epoch_rel_loss)

        print(
            f"epoch : {epoch+1}, "
            f"avg_loss : {avg_epoch_loss}, "
            f"avg_rec_loss : {avg_epoch_rec_loss}, "
            f"avg_fm_loss : {avg_epoch_fm_loss}, "
            f"avg_l1_loss : {avg_epoch_l1_loss}, "
            f"avg_mse_loss : {avg_epoch_mse_loss}, "
            f"avg_rel_loss : {avg_epoch_rel_loss}, "
            f"lambda_rec : {lambda_rec}, "
            f"lambda_losses : {lambda_losses}"
        )

        #Save model 
        if (epoch + 1) % 1 == 0:
            model_save_path = os.path.join(weights_dir, f"epoch_{epoch+1}_loss={avg_epoch_loss:.5f}.pt")
            torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(),"flow": flow.state_dict(),}, model_save_path)
        
        if (epoch + 1) % 4 == 0:
            latent_fm(
                model=flow,
                decoder=decoder,
                latent_dim=x_1.size(1),
                save_dir=images_dir,
                epoch=epoch + 1,
                training_mode=True,
                integration_steps=100,
                n_samples=4
            )  
        
    #Save losses 
    for key, values in loss_log_dict.items():
        path = os.path.join(save_dir, f"{key}.txt")
        with open(path, "w") as f:
            for value in values:
                f.write(f"{value}\n")

def latent_fm(model, decoder, latent_dim, save_dir, epoch, training_mode=False, integration_steps=100, n_samples=4): 
    model.eval()
    decoder.eval()

    cpu_device = torch.device("cpu")
    gpu_device = torch.device("mps")

    with torch.no_grad():
        x_t = torch.randn(n_samples, latent_dim, device=cpu_device)
        dt = 1.0 / integration_steps

        for i in range(integration_steps): 
            t = torch.full((n_samples, 1), fill_value=i/integration_steps, device=cpu_device)
            v = model(x_t, t)
            x_t = x_t + v * dt
        
        image = decoder(x_t.to(gpu_device))
        image = image.cpu()
    
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
    
    gpu_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    encoder = Encoder(3, LATENT_DIM).to(gpu_device)
    decoder = Decoder(3, LATENT_DIM).to(gpu_device)
    flow = Flow(latent_dim=LATENT_DIM, hidden_dim=128).to("cpu")
    ae_weights = torch.load("AUTOENCODER_MODEL-2026-08-15-19-25/weights/epoch_370_loss=0.00328.pt")
    flow_weights = torch.load("LatentFM_MODEL-2026-08-18-23-09/weights/epoch_297_loss=68.19995.pt")

    encoder.load_state_dict(ae_weights["encoder"])
    decoder.load_state_dict(ae_weights["decoder"])
    flow.load_state_dict(flow_weights)

    ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

    transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(), #[0,1]
                    transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
                    ])
    im_ds = ds[:]["image"]
    dataset = ImageToTensor(im_ds, transform)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)
    finetune(num_epochs = EPOCHS, flow = flow, encoder=encoder, decoder=decoder, dataloader=dataloader, B=B)