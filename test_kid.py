import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from torchmetrics.image.kid import KernelInceptionDistance as KID
from utils import ImageToTensor
from unet_vf import UNet_VF
from autoencoder import Decoder
from latent_flow_matching import Flow


def pixel_sample(model, n_samples, C, H, W, steps, device):
    model.eval()

    x = torch.randn(n_samples, C, H, W, device=device)
    dt = 1.0 / steps

    with torch.no_grad():
        for i in range(steps):
            t = torch.full((n_samples,), i / steps, device=device)
            x = x + model(x, t) * dt

    return ((x.clamp(-1, 1) + 1) / 2).cpu()


def latent_sample(flow, decoder, n_samples, latent_dim, steps):
    flow.eval()
    decoder.eval()

    z = torch.randn(n_samples, latent_dim)
    dt = 1.0 / steps

    with torch.no_grad():
        for i in range(steps):
            t = torch.full((n_samples, 1), i / steps)
            z = z + flow(z, t) * dt

        x = decoder(z)

    return (x.clamp(-1, 1) + 1) / 2


def test_kid(pixel_model, flow, decoder, dataset, batch_size, feature, subsets, subset_size, C, H, W, latent_dim, steps, device):
    kid_pixel = KID(feature=feature, subsets=subsets, subset_size=subset_size, normalize=True)
    kid_latent = KID(feature=feature, subsets=subsets, subset_size=subset_size, normalize=True)

    real = torch.stack([dataset[i] for i in range(len(dataset))])

    kid_pixel.update(real, real=True)
    kid_latent.update(real, real=True)

    n_samples = len(dataset)
    generated = 0

    while generated < n_samples:
        b = min(batch_size, n_samples - generated)

        pixel_batch = pixel_sample(pixel_model, b, C, H, W, steps, device)
        latent_batch = latent_sample(flow, decoder, b, latent_dim, steps)

        kid_pixel.update(pixel_batch, real=False)
        kid_latent.update(latent_batch, real=False)

        generated += b

    pixel_mean, pixel_std = kid_pixel.compute()
    latent_mean, latent_std = kid_latent.compute()

    print(f"Pixel KID  : {pixel_mean.item():.6f} ± {pixel_std.item():.6f}")
    print(f"Latent KID : {latent_mean.item():.6f} ± {latent_std.item():.6f}")


if __name__ == "__main__":
    torch.manual_seed(0)

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    C = 3
    H = 128
    W = 128

    TEST_BATCH_SIZE = 64

    KID_FEATURE = 2048
    SUBSETS = 100
    SUBSET_SIZE = 75

    PIXEL_T_EMB_DIM = 128

    LATENT_DIM = 1024
    FLOW_HIDDEN_DIM = 512
    LATENT_T_EMB_DIM = 128

    INTEGRATION_STEPS = 100

    pixel_model = UNet_VF(C, PIXEL_T_EMB_DIM).to(DEVICE)
    pixel_model.load_state_dict(torch.load("PIXEL.pt", map_location=DEVICE))

    decoder = Decoder(C, LATENT_DIM)
    flow = Flow(LATENT_DIM, FLOW_HIDDEN_DIM, LATENT_T_EMB_DIM)

    ae_weights = torch.load("AUTOENCODER.pt", map_location="cpu")
    flow_weights = torch.load("FLOW.pt", map_location="cpu")

    decoder.load_state_dict(ae_weights["decoder"])
    flow.load_state_dict(flow_weights)

    ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train[800:]")

    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor()
    ])

    dataset = ImageToTensor(ds[:]["image"], transform)

    test_kid(pixel_model, flow, decoder, dataset, TEST_BATCH_SIZE, KID_FEATURE, SUBSETS, SUBSET_SIZE, C, H, W, LATENT_DIM, INTEGRATION_STEPS, DEVICE)