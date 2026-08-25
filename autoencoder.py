import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from datasets import load_dataset
import datetime
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
    """
    Encoder: from a 3x128x128 input we arrive to a 512x4x4 embedding
    Flow: 
                Augment Channels -> Residual Block -> Downsample

    assuming input image is 3x128x128, by iterating over formula
                 O = f(I, K, P, S) = floor( (I - K + 2P) / S ) + 1 
    where O = output dim, I = input dim, K = kernel dim, P = padding, S = stride
    we obtain a bottleneck layer of dimensions O6 = f(8, 3, 1, 2) = 4, C = 512
    """
    def __init__(
        self, 
        n_channels = 3, 
        latent_dim = 512, 
        hidden_dims = None
    ):
        super().__init__()
        
        if hidden_dims is None: 
            hidden_dims = [32, 64, 128, 256, 512]

        modules = []

        self.n_channels = n_channels
        for h in hidden_dims: 
            #Learn features
            modules.append(
                nn.Sequential(
                    nn.Conv2d(n_channels, out_channels=h, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(8,h),
                    nn.SiLU()
                )
            )
            #ResBlock
            modules.append(ResBlock(n_channels=h))
            
            #Downsample
            modules.append(
                nn.Sequential(
                    nn.Conv2d(h, out_channels=h, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(8,h),
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
    def __init__(
        self,
        n_channels=3,
        latent_dim=512,
        hidden_dims=None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        hidden_dims = hidden_dims[::-1]
        # [512, 256, 128, 64, 32]

        self.flatten_dim = hidden_dims[0] * 4 * 4

        self.projection = nn.Linear(
            latent_dim,
            self.flatten_dim
        )

        self.unflatten = nn.Unflatten(
            1,
            (hidden_dims[0], 4, 4)
        )

        modules = []

        for i in range(len(hidden_dims) - 1):

            # Upsample
            modules.append(nn.Upsample(scale_factor=2,mode="nearest"))

            # Learn features
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ),
                    nn.GroupNorm(
                        8,
                        hidden_dims[i + 1]
                    ),
                    nn.SiLU()
                )
            )

            # Residual processing
            modules.append(ResBlock(n_channels=hidden_dims[i + 1]))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="nearest"
            ),
            nn.Conv2d(
                hidden_dims[-1],
                n_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x
    
def train(num_epochs, encoder, decoder, dataloader):
    #Save dir
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_dir = f"AUTOENCODER_MODEL-{timestamp}"
    weights_dir = os.path.join(save_dir, "weights")
    os.makedirs(weights_dir)
    print(f"Training output directory : {save_dir}")

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        for batch in dataloader:
            z = encoder(batch)
            recons = decoder(z)

            #loss = F.mse_loss(recons, batch)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"epoch : {epoch+1}, avg_loss : {avg_epoch_loss}")
        epoch_loss = 0.0

        #Save model 
        if (epoch + 1) % 10 == 0:
            model_save_path = os.path.join(weights_dir, f"epoch_{epoch+1}_loss={avg_epoch_loss:.5f}.pt")
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            }, model_save_path)

def test(encoder, decoder, dataloader):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        batch = next(iter(dataloader))

        z = encoder(batch)
        recons = decoder(z)
        print(f"batch shape : {batch.shape}, z shape : {z.shape}, recons.shape : {recons.shape}")

        save_image(
            batch,
            "original.png",
            normalize=True,
            value_range=(-1, 1)
        )

        save_image(
            recons,
            "reconstructed.png",
            normalize=True,
            value_range=(-1, 1)
        )

    print("Test images saved.")

if __name__ == "__main__":
    encoder = Encoder(3, 512)
    decoder = Decoder(3, 512)
    
    weights = torch.load("AUTOENCODER_MODEL-2026-08-15-00-20/weights/epoch_600_loss=0.00448.pt")
    encoder.load_state_dict(weights["encoder"])
    decoder.load_state_dict(weights["decoder"])

    B=16
    ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(), #[0,1]
                    transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
                    ])
    im_ds = ds[:]["image"]
    dataset = ImageToTensor(im_ds, transform)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)

    train(600, encoder, decoder, dataloader)
    test(encoder, decoder, dataloader)
        
