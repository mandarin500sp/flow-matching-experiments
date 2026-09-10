import matplotlib.pyplot as plt
import argparse


def read_losses(file_path):
    with open(file_path, "r") as f:
        losses = [float(line.strip()) for line in f if line.strip()]

    return losses


def plot_loss(loss_file, label, color):
    losses = read_losses(loss_file)
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(6, 6))

    plt.plot(epochs, losses, color=color, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("loss_file", type=str)
    parser.add_argument("label", type=str)
    parser.add_argument("color", type=str)

    args = parser.parse_args()

    plot_loss(args.loss_file, args.label, args.color)