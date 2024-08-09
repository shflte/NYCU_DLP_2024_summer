import os
import torch
from tqdm import tqdm
from utils import (
    load_config,
    kl_criterion,
    kl_annealing,
    teacher_forcing,
    set_random_seed,
    show_loss,
    show_loss_kl_anneal,
)
from dataloader import get_dataloader
from modules.vae_model import VAE_Model


def save_checkpoint(model, optimizer, scheduler, model_root, epoch):
    checkpoint_dir = os.path.join(model_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        },
        os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"),
    )
    print(f"Checkpoint saved at epoch {epoch}")


def save_final_model(model, model_root):
    final_model_path = os.path.join(model_root, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")


def load_latest_checkpoint(model, optimizer, scheduler, model_root):
    checkpoint_dir = os.path.join(model_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if files:
        latest_checkpoint = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, latest_checkpoint), weights_only=True
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]
    return -1


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

    # t: frame_t to generate
    for t in range(1, T):
        img = images[t - 1] if adapt_TeacherForcing or last_pred is None else last_pred

        # encode
        img_features = model.frame_transformation(img)
        real_frame_features = model.frame_transformation(images[t])
        label_features = model.label_transformation(labels[t])

        # gaussian predictor
        z, mu, logvar = model.Gaussian_Predictor(real_frame_features, label_features)

        # decode
        output = model.Decoder_Fusion(img_features, label_features, z)

        # generate
        prediction = model.Generator(output)
        last_pred = prediction

        # loss
        mse_loss = mse_criterion(prediction, images[t])
        kl_loss = kl_criterion(mu, logvar, B)
        loss = mse_loss + kl_anneal_beta * kl_loss
        total_loss += loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    return total_loss


def train(args):
    # set random seed
    set_random_seed(args.seed)

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
        shuffle=False,
        drop_last=True,
    )

    # model
    os.makedirs(args.model_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2, 5], gamma=0.1
    )

    # load latest checkpoint if available
    current_epoch = (
        load_latest_checkpoint(model, optimizer, scheduler, args.model_root) + 1
    )

    # loss
    mse_criterion = torch.nn.MSELoss()

    # KL annealing
    kl_anneal = kl_annealing(args, current_epoch)

    # Teacher Forcing Ratio
    tf = teacher_forcing(args, current_epoch)

    # training
    model.train()
    epochs = args.num_epoch if not args.fast_train else args.fast_train_epoch
    loss_list = []
    kl_beta_list = []
    for epoch in range(current_epoch, epochs):
        adapt_teacher_forcing = tf.adapt_teacher_forcing()
        kl_anneal_beta = kl_anneal.get_beta()
        kl_beta_list.append(kl_anneal_beta)

        progress_bar = tqdm(train_loader, ncols=120)
        total_loss_per_epoch = 0.0
        for img, label in progress_bar:
            img = img.to(args.device)
            label = label.to(args.device)
            loss = train_step(
                model,
                img,
                label,
                adapt_teacher_forcing,
                optimizer,
                mse_criterion,
                kl_anneal_beta,
            )
            total_loss_per_epoch += loss.item()
            progress_bar.set_postfix(
                {
                    "Epoch": epoch,
                    "Loss": loss.item(),
                    "KL Beta": kl_anneal_beta,
                    "TF": str(adapt_teacher_forcing),
                    "TFR": f"{tf.get_tfr():.2f}",
                }
            )

        kl_anneal.update()
        tf.update()

        if epoch % args.per_save == 0:
            save_checkpoint(model, optimizer, scheduler, args.model_root, epoch)

        scheduler.step()

        loss_list.append(total_loss_per_epoch / len(train_loader))

    # Save final model
    save_final_model(model, args.model_root)

    # Show the result
    show_loss(loss_list)
    show_loss_kl_anneal(loss_list, kl_beta_list)


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
