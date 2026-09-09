import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from utils import ImageToTensor
from unet_vf import UNet_VF
from autoencoder import Encoder
from latent_flow_matching import Flow


def pixel_loss(model, dataloader, device):
    model.eval()

    total_error_energy = 0.0
    total_target_energy = 0.0

    with torch.no_grad():
        for x_1 in dataloader:
            x_1 = x_1.to(device)

            x_0 = torch.randn_like(x_1)

            t = torch.rand(x_1.size(0), device=device)
            tt = t[:, None, None, None]

            x_t = (1 - tt) * x_0 + tt * x_1
            u_t = x_1 - x_0

            v_pred = model(x_t, t)

            error_energy = (v_pred - u_t).square().sum()
            target_energy = u_t.square().sum()

            total_error_energy += error_energy.item()
            total_target_energy += target_energy.item()

    return total_error_energy / total_target_energy


def latent_loss(flow, encoder, dataloader, device):
    flow.eval()
    encoder.eval()

    total_error_energy = 0.0
    total_target_energy = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            x_1 = encoder(batch)
            x_0 = torch.randn_like(x_1)

            t = torch.rand(x_1.size(0), 1, device=device)

            x_t = (1 - t) * x_0 + t * x_1
            u_t = x_1 - x_0

            v_pred = flow(x_t, t)

            error_energy = (v_pred - u_t).flatten(1).square().sum()
            target_energy = u_t.flatten(1).square().sum()

            total_error_energy += error_energy.item()
            total_target_energy += target_energy.item()

    return total_error_energy / total_target_energy


def main(pixel_model, latent_flow, encoder, dataloader, device, n_repeats):
    pixel_losses = []
    latent_losses = []

    for _ in range(n_repeats):
        pixel_losses.append(pixel_loss(pixel_model, dataloader, device))
        latent_losses.append(latent_loss(latent_flow, encoder, dataloader, device))

    pixel_losses = torch.tensor(pixel_losses)
    latent_losses = torch.tensor(latent_losses)

    print("Pixel mean:", pixel_losses.mean().item())
    print("Pixel std:", pixel_losses.std().item())
    print("Latent mean:", latent_losses.mean().item())
    print("Latent std:", latent_losses.std().item())


if __name__ == "__main__":
    DEVICE = torch.device("mps")

    C = 3
    H = 128
    W = 128

    TEST_BATCH_SIZE = 64
    N_REPEATS = 10

    PIXEL_T_EMB_DIM = 128

    LATENT_DIM = 1024
    FLOW_HIDDEN_DIM = 512
    LATENT_T_EMB_DIM = 128

    pixel_model = UNet_VF(C, PIXEL_T_EMB_DIM).to(DEVICE)
    latent_flow = Flow(LATENT_DIM, FLOW_HIDDEN_DIM, LATENT_T_EMB_DIM).to(DEVICE)
    encoder = Encoder(C, LATENT_DIM).to(DEVICE)

    pixel_model.load_state_dict(torch.load("PIXEL.pt", map_location=DEVICE))
    latent_flow.load_state_dict(torch.load("FLOW.pt", map_location=DEVICE))

    ae_weights = torch.load("AUTOENCODER.pt", map_location=DEVICE)
    encoder.load_state_dict(ae_weights["encoder"])

    ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train[:800]")

    transform = transforms.Compose([
                transforms.Resize((H, W)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ])

    dataset = ImageToTensor(ds[:]["image"], transform)
    dataloader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    main(pixel_model, latent_flow, encoder, dataloader, DEVICE, N_REPEATS)