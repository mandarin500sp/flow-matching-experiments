import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from utils import ImageToTensor
from autoencoder import Encoder, Decoder


def reconstruction_loss(encoder, decoder, dataloader, device):
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            z = encoder(batch)
            recons = decoder(z)

            loss = F.mse_loss(recons, batch)

            total_loss += loss.item() * batch.size(0)
            total_samples += batch.size(0)

    return total_loss / total_samples


def main(encoder, decoder, dataloader, device):
    loss = reconstruction_loss(encoder, decoder, dataloader, device)
    print("Train reconstruction MSE:", loss)


if __name__ == "__main__":
    DEVICE = torch.device("mps")

    C = 3
    H = 128
    W = 128

    BATCH_SIZE = 64
    LATENT_DIM = 1024

    encoder = Encoder(C, LATENT_DIM).to(DEVICE)
    decoder = Decoder(C, LATENT_DIM).to(DEVICE)

    ae_weights = torch.load("AUTOENCODER.pt", map_location=DEVICE)

    encoder.load_state_dict(ae_weights["encoder"])
    decoder.load_state_dict(ae_weights["decoder"])

    ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train[800:]")

    transform = transforms.Compose([
                transforms.Resize((H, W)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ])

    dataset = ImageToTensor(ds[:]["image"], transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    main(encoder, decoder, dataloader, DEVICE)