import yaml
import torch
import torchvision
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from matplotlib import pyplot as plt
from utils import set_random_seed, show_losses
from dataset import CLEVRDataset
from models.conditional_unet import ClassConditionedUnet


def train(config):
    # Dataset
    dataset = CLEVRDataset(
        dataset_dir=config["dataset_dir"],
        train_json=config["train_json_path"],
        objects_json=config["objects_json_path"],
    )
    train_dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True
    )

    # Model
    model = ClassConditionedUnet(num_classes=config["num_classes"])
    model = model.cuda()

    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    losses = []
    for epoch in range(config["epochs"]):
        model.train()
        for images, labels in tqdm(train_dataloader):
            # Prepare data
            images = images.cuda()
            labels = labels.cuda()

            # Add noise
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, 999, (images.shape[0],)).long().cuda()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # Forward pass
            predictions = model(noisy_images, timesteps, labels)
            loss = criterion(predictions, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save loss value
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        print(f"Finished epoch {epoch}. Loss: {avg_loss:.5f}")

    # Plot the loss values
    show_losses(losses)

    # Save the model
    torch.save(model.state_dict(), config["model_path"])


if __name__ == "__main__":
    # Load arguments from config file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    set_random_seed(config["seed"])

    train(config)
