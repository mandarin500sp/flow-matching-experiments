import os
import torch
import matplotlib.pyplot as plt

from unet_vf import UNet_VF
from autoencoder import Decoder
from latent_flow_matching import Flow


def pixel_trajectory(model, C, H, W, steps, checkpoints, device):
    model.eval()

    torch.manual_seed(149)
    x = torch.randn(1, C, H, W, device=device)

    images = {0: (x.clamp(-1, 1) + 1) / 2}
    dt = 1.0 / steps

    with torch.no_grad():
        for i in range(steps):
            t = torch.full((1,), i/steps, device=device)
            x = x + model(x, t) * dt

            if i + 1 in checkpoints:
                images[i + 1] = (x.clamp(-1, 1) + 1) / 2

    return images


def latent_trajectory(flow, decoder, latent_dim, steps, checkpoints):
    flow.eval()
    decoder.eval()

    torch.manual_seed(12)
    z = torch.randn(1, latent_dim)

    with torch.no_grad():
        images = {0: (decoder(z).clamp(-1, 1) + 1) / 2}
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((1, 1), i/steps)
            z = z + flow(z, t) * dt

            if i + 1 in checkpoints:
                images[i + 1] = (decoder(z).clamp(-1, 1) + 1) / 2

    return images


def compare_trajectory(pixel_model, flow_model, decoder, C, H, W, latent_dim, checkpoints, steps, pixel_device, save_path):
    pixel_images = pixel_trajectory(pixel_model, steps, checkpoints, pixel_device)
    latent_images = latent_trajectory(flow_model, decoder, steps, checkpoints)

    fig, axes = plt.subplots(2, len(checkpoints), figsize=(2.5 * len(checkpoints), 5))

    for j, step in enumerate(checkpoints):
        pixel = pixel_images[step][0].permute(1, 2, 0).cpu().numpy()
        latent = latent_images[step][0].permute(1, 2, 0).cpu().numpy()

        axes[0, j].imshow(pixel)
        axes[1, j].imshow(latent)

        axes[0, j].set_title(f"{step}", fontsize=11)

        axes[0, j].axis("off")
        axes[1, j].axis("off")

    fig.text(0.035, 0.70, "Pixel", fontsize=11, ha="center", va="center", rotation=90)
    fig.text(0.035, 0.30, "Latent", fontsize=11, ha="center", va="center", rotation=90)

    plt.subplots_adjust(left=0.055, right=0.995, top=0.99, bottom=0.01, wspace=0.03, hspace=0.03)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    pixel_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    C, H, W = 3, 128, 128
    T_EMB_DIM = 128
    LATENT_DIM = 1024
    FLOW_HIDDEN_DIM = 512

    STEPS = 100
    CHECKPOINTS = [0, 25, 50, 75, 100]

    pixel_model = UNet_VF(C, T_EMB_DIM).to(pixel_device)
    pixel_model.load_state_dict(torch.load("PIXEL.pt", map_location=pixel_device))

    decoder = Decoder(C, LATENT_DIM)
    ae_weights = torch.load("AUTOENCODER.pt", map_location="cpu")
    decoder.load_state_dict(ae_weights["decoder"])

    flow_model = Flow(LATENT_DIM, FLOW_HIDDEN_DIM)
    flow_model.load_state_dict(torch.load("LATENT.pt", map_location="cpu"))

    compare_trajectory(pixel_model, flow_model, decoder, C, H, W, LATENT_DIM, CHECKPOINTS, STEPS, pixel_device, "trajectory.png")