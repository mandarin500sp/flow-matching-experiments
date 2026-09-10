import matplotlib.pyplot as plt
import argparse


def read_losses(file_path):
    with open(file_path, "r") as f:
        losses = [float(line.strip()) for line in f if line.strip()]

    return losses


def plot_losses(loss_file, rel_loss_file):
    losses = read_losses(loss_file)
    rel_losses = read_losses(rel_loss_file)

    loss_epochs = range(1, len(losses) + 1)
    rel_loss_epochs = range(1, len(rel_losses) + 1)

    plt.figure(figsize=(6, 6))

    plt.plot(loss_epochs, losses, color="orange", label="Pixel FM Relative Loss")
    plt.plot(rel_loss_epochs, rel_losses, color="green", label="Latent FM Relative Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("loss_file", type=str)
    parser.add_argument("rel_loss_file", type=str)

    args = parser.parse_args()

    plot_losses(args.loss_file, args.rel_loss_file)