import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from modules.vae_model import VAE_Model
from dataloader import get_dataloader
from utils import load_config, save_submission, make_gif


def test_step(model, img, label, device, idx, save_root):
    img = img.permute(1, 0, 2, 3, 4)  # change tensor into (seq, B, C, H, W)
    label = label.permute(1, 0, 2, 3, 4)  # change tensor into (seq, B, C, H, W)
    assert label.shape[0] == 630, "Testing pose sequence should be 630"
    assert img.shape[0] == 1, "Testing video sequence should be 1"

    decoded_frame_list = [img[0].cpu()]
    label_list = []

    # TODO: Implement the inference logic
    # Your inference logic here

    # Please do not modify this part, it is used for visualization
    generated_frame = torch.stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
    label_frame = torch.stack(label_list).permute(1, 0, 2, 3, 4)

    assert generated_frame.shape == (
        1,
        630,
        3,
        32,
        64,
    ), f"The shape of output should be (1, 630, 3, 32, 64), but your output shape is {generated_frame.shape}"

    make_gif(generated_frame[0], os.path.join(save_root, f"pred_seq{idx}.gif"))

    generated_frame = generated_frame.reshape(630, -1)
    return generated_frame


def test(args):
    os.makedirs(args.save_root, exist_ok=True)

    # Load model
    model = VAE_Model(args).to(args.device)
    if args.ckpt_path:
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    # Load data
    test_loader = get_dataloader(
        root=args.DR,
        frame_H=args.frame_H,
        frame_W=args.frame_W,
        mode="val",
        video_len=args.val_vi_len,
        batch_size=1,
        num_workers=args.num_workers,
        partial=1.0,
        shuffle=False,
        drop_last=True,
    )

    model.eval()
    pred_seq_list = []
    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(test_loader, ncols=80)):
            img = img.to(args.device)
            label = label.to(args.device)
            pred_seq = test_step(model, img, label, args.device, idx, args.save_root)
            pred_seq_list.append(pred_seq)

    save_submission(pred_seq_list, args.save_root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/test.yaml", help="Path to the config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    test(args)
