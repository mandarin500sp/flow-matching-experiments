import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from utils import ImageToTensor, get_time_embeddings
from autoencoder import Encoder, Decoder


class Flow(nn.Module):
    """
    The parametrizing the vector field. As in the UNet-VF model, we use a time embedding to enhance training performance.
    Once the flow is learned, one can integrate it over [0,1] with initial codition x0 to generate new samples.
    """
    def __init__(self, latent_dim, hidden_dim, time_emb_dim=64):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x_t, t):
        t_emb = get_time_embeddings(t.squeeze(-1), self.time_emb_dim, scale_factor=10)
        t_emb = self.time_mlp(t_emb)

        return self.net(torch.cat((x_t, t_emb), dim=-1))


def main(dataloader=None, B=None, epochs=None, train=True, n_samples=None, flow=None, encoder=None, decoder=None, latent_dim=1024, hidden_dim=512, time_emb_dim=64, lr=1e-3, save_model_every=10, integration_steps=100):
    if train:
        #Create training output dirs
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        save_dir = f"LatentFM-{timestamp}"
        weights_dir = os.path.join(save_dir, "weights")
        images_dir = os.path.join(save_dir, "images")
        os.makedirs(weights_dir)
        os.makedirs(images_dir)

        #Save training parameters
        with open(os.path.join(save_dir, "training_params.txt"), "w") as f:
            f.write(f"EPOCHS : {epochs}\n")
            f.write(f"BATCH_SIZE : {B}\n")
            f.write(f"LATENT_DIM : {latent_dim}\n")
            f.write(f"FLOW_HIDDEN_DIM : {hidden_dim}\n")
            f.write(f"T_EMB_DIM : {time_emb_dim}\n")
            f.write(f"LEARNING_RATE : {lr}\n")
            f.write(f"INTEGRATION_STEPS : {integration_steps}\n")

        if flow is None:
            flow = Flow(latent_dim, hidden_dim, time_emb_dim)

        loss_log = train_FM(epochs, flow, encoder, decoder, dataloader, lr, weights_dir, images_dir, n_samples, save_model_every, integration_steps)

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
        images_dir = f"LatentFM_TEST-{timestamp}"
        os.makedirs(images_dir)

        latent_fm(flow, decoder, latent_dim, images_dir, integration_steps, n_samples)


def train_FM(epochs, flow, encoder, decoder, dataloader, lr, weights_dir, images_dir, n_samples, save_model_every, integration_steps):
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    """
    Trains the latent Flow. 
    """

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    #Loss log dictionary
    loss_log = {
        "epoch_losses": [],
        "epoch_rel_losses": []
    }

    for epoch in range(epochs):
        flow.train()

        epoch_loss = 0.0
        epoch_error_energy = 0.0
        epoch_target_energy = 0.0

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
            v_pred = flow(x_t, t)

            loss = F.mse_loss(v_pred, dx_t)

            #Computation of the relative loss
            with torch.no_grad():
                #accumulate error energies
                error_energy = (v_pred - dx_t).flatten(1).square().sum()

                #accumulate target energies
                target_energy = dx_t.flatten(1).square().sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Loss logging
            epoch_loss += loss.item()
            epoch_error_energy += error_energy.item()
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
            torch.save(flow.state_dict(), model_save_path)

            latent_fm(flow, decoder, x_1.size(1), images_dir, integration_steps, n_samples, True, epoch+1)

    return loss_log


def latent_fm(model, decoder, latent_dim, save_dir, integration_steps=100, n_samples=4, training_mode=False, epoch=None):
    """
    Integrates the vector field in [0,1] with initial condition x0 ~ N(0,I) using the Euler method. Outputs the
    generated image.
    """
    model.eval()
    decoder.eval()

    with torch.no_grad():
        x_t = torch.randn(n_samples, latent_dim)
        dt = 1.0 / integration_steps

        #Euler method
        for i in range(integration_steps):
            t = torch.full((n_samples, 1), fill_value=i/integration_steps)
            v = model(x_t, t)
            x_t = x_t + v * dt

        image = decoder(x_t)
        out = (image.clamp(-1, 1) + 1) / 2  # map back to [0,1] for saving

        for i in range(n_samples):
            if training_mode:
                img_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{i}.png")
            else:
                img_path = os.path.join(save_dir, f"sample_{i}.png")

            torchvision.utils.save_image(out[i], img_path)

        return out


if __name__ == "__main__":
    B = 16
    EPOCHS = 300
    LATENT_DIM = 1024
    FLOW_HIDDEN_DIM = 512
    T_EMB_DIM = 128
    LR = 1e-3
    SAVE_MODEL_EVERY = 5
    N_SAMPLES = 4
    INTEGRATION_STEPS = 100

    encoder = Encoder(3, LATENT_DIM)
    decoder = Decoder(3, LATENT_DIM)
    flow = Flow(LATENT_DIM, FLOW_HIDDEN_DIM, T_EMB_DIM)

    ae_weights = torch.load("AUTOENCODER.pt")
    #flow_weights = torch.load("FLOW.pt")

    encoder.load_state_dict(ae_weights["encoder"])
    decoder.load_state_dict(ae_weights["decoder"])
    #flow.load_state_dict(flow_weights)

    ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train[:800]")

    transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(), #[0,1]
                transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
                ])

    dataset = ImageToTensor(ds[:]["image"], transform)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)

    main(dataloader, B, EPOCHS, True, N_SAMPLES, flow, encoder, decoder, LATENT_DIM, FLOW_HIDDEN_DIM, T_EMB_DIM, LR, SAVE_MODEL_EVERY, INTEGRATION_STEPS)
