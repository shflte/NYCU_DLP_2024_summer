import os
import yaml
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from utils import set_random_seed, show_losses
from dataset import CLEVRDataset, CLEVRDatasetEval
from evaluator import evaluation_model

from models.conditional_unet import ClassConditionedUnet


def load_checkpoint(model, optimizer, checkpoint_dir="checkpoints"):
    """
    Load the latest checkpoint if available.

    Args:
        model (nn.Module): The model to load the state_dict into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state_dict into.
        checkpoint_dir (str): Directory where checkpoints are saved.

    Returns:
        int: The epoch number to start training from.
    """
    if not os.path.exists(checkpoint_dir):
        return 0
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        print("No checkpoints found.")
        return 0

    latest_checkpoint = max(
        checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    return start_epoch


def save_checkpoint(model, optimizer, epoch, checkpoint_dir="checkpoints"):
    """
    Save the current model and optimizer state.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        checkpoint_dir (str): Directory where checkpoints are saved.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved: {checkpoint_path}")


def validate(model, config):
    # Load the evaluation dataset
    val_dataset = CLEVRDatasetEval(
        eval_json=config["eval_json_path"],
        objects_json=config["objects_json_path"],
    )
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    # Evaluation model
    evaluator = evaluation_model()

    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["denoise_steps"], beta_schedule="squaredcos_cap_v2"
    )

    # Generate images and evaluate
    model.eval()
    total_acc = 0
    total_samples = 0
    for labels in val_dataloader:
        labels = labels.cuda()
        x = torch.randn((32, 3, 64, 64)).cuda()
        for t in range(config["denoise_steps"]):
            with torch.no_grad():
                residual = model(x, t, labels)
            x = noise_scheduler.step(residual, t, x).prev_sample

        # Evaluate the generated images
        acc = evaluator.eval(x, labels)

        # Accumulate accuracy
        batch_size = labels.size(0)
        total_acc += acc * batch_size
        total_samples += batch_size

    # Calculate the final average accuracy
    final_acc = total_acc / total_samples
    return round(final_acc, 3)


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

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Load checkpoint if available
    start_epoch = load_checkpoint(model, optimizer, checkpoint_dir="checkpoints")

    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["denoise_steps"], beta_schedule="squaredcos_cap_v2"
    )

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    losses = []
    accuracies = []
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        for images, labels in tqdm(train_dataloader):
            # Prepare data
            images = images.cuda()
            labels = labels.cuda()

            # Add noise
            noise = torch.randn_like(images)
            timesteps = (
                torch.randint(0, config["denoise_steps"] - 1, (images.shape[0],))
                .long()
                .cuda()
            )
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

        # Validate the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            acc = validate(model, config)
            accuracies.append(acc)
            print(f"Validation accuracy: {acc}")

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, checkpoint_dir=config["checkpoint_dir"]
            )

    # Plot the loss values
    show_losses(losses)

    # Save the final model
    torch.save(model.state_dict(), config["model_path"])


if __name__ == "__main__":
    # Load arguments from config file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    set_random_seed(config["seed"])

    train(config)
