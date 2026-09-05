import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import datetime

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from datasets import load_dataset

from utils import ImageToTensor


class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.res = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.GroupNorm(8, n_channels),
            nn.SiLU(),
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.GroupNorm(8, n_channels)
        )

        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.res(x))


class Encoder(nn.Module):
    def __init__(self, n_channels=3, latent_dim=512, hidden_dims=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        modules = []

        self.n_channels = n_channels

        for h in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(n_channels, h, 3, 1, 1),
                    nn.GroupNorm(8, h),
                    nn.SiLU()
                )
            )

            modules.append(ResBlock(h))

            modules.append(
                nn.Sequential(
                    nn.Conv2d(h, h, 3, 2, 1),
                    nn.GroupNorm(8, h),
                    nn.SiLU()
                )
            )

            n_channels = h

        self.encoder = nn.Sequential(*modules)
        self.flatten = nn.Flatten()
        self.flatten_dim = hidden_dims[-1] * 4 * 4
        self.latent_proj = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.latent_proj(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_channels=3, latent_dim=512, hidden_dims=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        hidden_dims = hidden_dims[::-1]

        self.flatten_dim = hidden_dims[0] * 4 * 4
        self.projection = nn.Linear(latent_dim, self.flatten_dim)

        self.unflatten = nn.Unflatten(
            1,
            (hidden_dims[0], 4, 4)
        )

        modules = []

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Upsample(scale_factor=2, mode="nearest"))

            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], 3, 1, 1),
                    nn.GroupNorm(8, hidden_dims[i + 1]),
                    nn.SiLU()
                )
            )

            modules.append(ResBlock(hidden_dims[i + 1]))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_dims[-1], n_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x


def train(num_epochs, encoder, decoder, dataloader, lr, save_model_every, device):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_dir = f"AUTOENCODER_MODEL-{timestamp}"
    weights_dir = os.path.join(save_dir, "weights")

    os.makedirs(weights_dir)

    loss_file = os.path.join(save_dir, "loss.txt")

    print(f"Training output directory : {save_dir}")

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()

        epoch_loss = 0.0

        for batch in dataloader:
            batch = batch.to(device)

            z = encoder(batch)
            recons = decoder(z)

            loss = F.mse_loss(recons, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)

        print(f"epoch : {epoch+1}, avg_loss : {avg_epoch_loss}")

        with open(loss_file, "a") as f:
            f.write(f"{avg_epoch_loss}\n")

        if (epoch + 1) % save_model_every == 0:
            model_save_path = os.path.join(weights_dir, f"epoch_{epoch+1}_loss={avg_epoch_loss:.5f}.pt")

            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            }, model_save_path)


def test(encoder, decoder, dataloader, device):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        batch = next(iter(dataloader))
        batch = batch.to(device)

        z = encoder(batch)
        recons = decoder(z)

        print(f"batch shape : {batch.shape}, z shape : {z.shape}, recons.shape : {recons.shape}")

        save_image(
            batch.cpu(),
            "original.png",
            normalize=True,
            value_range=(-1, 1)
        )

        save_image(
            recons.cpu(),
            "reconstructed.png",
            normalize=True,
            value_range=(-1, 1)
        )

    print("Test images saved.")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    C = 3
    H = 128
    W = 128

    LATENT_DIM = 1024
    B = 16

    NUM_EPOCHS = 600
    LR = 1e-3
    SAVE_MODEL_EVERY = 15

    encoder = Encoder(C, LATENT_DIM).to(device)
    decoder = Decoder(C, LATENT_DIM).to(device)

    ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train[:800]")

    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    im_ds = ds[:]["image"]
    dataset = ImageToTensor(im_ds, transform)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)

    train(NUM_EPOCHS, encoder, decoder, dataloader, LR, SAVE_MODEL_EVERY, device)