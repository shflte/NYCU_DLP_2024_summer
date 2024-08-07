import os
import torch
from tqdm import tqdm
import random
from utils import load_config, kl_criterion, kl_annealing
from dataloader import get_dataloader
from modules.vae_model import VAE_Model


def save_checkpoint(model, optimizer, model_root, epoch):
    checkpoint_dir = os.path.join(model_root, "checkpoints")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"),
    )
    print(f"Checkpoint saved at epoch {epoch}")


def save_final_model(model, optimizer, model_root):
    final_model_path = os.path.join(model_root, "final_model.pth")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        final_model_path,
    )
    print(f"Final model saved at {final_model_path}")


def train_step(
    model,
    images,
    labels,
    adapt_TeacherForcing,
    optimizer,
    mse_criterion,
    kl_anneal_beta,
):
    B, T, C, H, W = images.shape
    images = images.view(T, B, C, H, W)
    labels = labels.view(T, B, C, H, W)
    last_pred = None
    total_loss = 0.0
    for t in range(T - 1):
        img = images[t] if adapt_TeacherForcing and last_pred is None else last_pred
        img_features = model.frame_transformation(img)
        label_features = model.label_transformation(labels[t])

        z, mu, logvar = model.Gaussian_Predictor(img_features, label_features)
        output = model.Decoder_Fusion(img_features, label_features, z)
        output = model.Generator(output)
        last_pred = output

        loss = mse_criterion(output, images[t + 1]) + kl_anneal_beta * kl_criterion(
            mu, logvar, B
        )
        total_loss += loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss


def train(args):
    # dataset
    train_loader = get_dataloader(
        root=args.dataset_root,
        frame_H=args.frame_H,
        frame_W=args.frame_W,
        mode="train",
        video_len=args.train_vi_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        partial=args.fast_partial if args.fast_train else args.partial,
        shuffle=True,
        drop_last=True,
    )

    # model
    os.makedirs(args.model_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    checkpoint_dir = os.path.join(args.model_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if files:
        latest_checkpoint = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        final_model = torch.load(os.path.join(checkpoint_dir, latest_checkpoint))
        model.load_state_dict(final_model["state_dict"])
        current_epoch = final_model["epoch"]
    else:
        current_epoch = 0

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2, 5], gamma=0.1
    )

    # loss
    mse_criterion = torch.nn.MSELoss()

    # KL annealing
    kl_anneal = kl_annealing(args, current_epoch)

    # training
    model.train()
    for epoch in range(current_epoch, args.num_epoch):
        adapt_TeacherForcing = random.random() < args.tfr
        kl_anneal.update()
        kl_anneal_beta = kl_anneal.get_beta()

        progress_bar = tqdm(train_loader, ncols=120)
        for img, label in progress_bar:
            img = img.to(args.device)
            label = label.to(args.device)
            loss = train_step(
                model,
                img,
                label,
                adapt_TeacherForcing,
                optimizer,
                mse_criterion,
                kl_anneal_beta,
            )
            progress_bar.set_postfix(
                {"Epoch": epoch, "Loss": loss.item(), "KL Beta": kl_anneal_beta}
            )

        if epoch % args.per_save == 0:
            save_checkpoint(model, optimizer, args.model_root, epoch)

        scheduler.step()

        # Update Teacher Forcing Ratio
        if epoch >= args.tfr_sde:
            args.tfr = max(args.tfr_min, args.tfr - args.tfr_d_step)
            print(f"Epoch {epoch}: Teacher Forcing Ratio updated to {args.tfr}")

    # Save final model
    save_final_model(model, optimizer, args.model_root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/train.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    train(args)
