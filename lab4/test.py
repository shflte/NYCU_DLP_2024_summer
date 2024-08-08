import os
import torch
from tqdm import tqdm
import numpy as np
from modules.vae_model import VAE_Model
from dataloader import get_dataloader
from utils import load_config, save_submission, make_gif


def test_step(model, images, labels, device, idx, pred_root):
    B, T, C, H, W = labels.shape
    images = images.permute(1, 0, 2, 3, 4)
    labels = labels.permute(1, 0, 2, 3, 4)
    assert labels.shape[0] == 630, "Testing pose sequence should be 630"
    assert images.shape[0] == 1, "Testing video sequence should be 1"
    decoded_frame_list = [images[0].cpu()]

    last_pred = images[0].to(device)
    # t: frame t to predict
    for t in range(1, T):
        img_features = model.frame_transformation(last_pred)
        label_features = model.label_transformation(labels[t])

        z = torch.randn(B, 12, H, W).to(device)
        output = model.Decoder_Fusion(img_features, label_features, z)
        output = model.Generator(output)
        last_pred = output

        decoded_frame_list.append(output.cpu())

    # Please do not modify this part, it is used for visualization
    generated_frame = torch.stack(decoded_frame_list).permute(1, 0, 2, 3, 4)

    assert generated_frame.shape == (
        1,
        630,
        3,
        32,
        64,
    ), f"The shape of output should be (1, 630, 3, 32, 64), but your output shape is {generated_frame.shape}"

    make_gif(generated_frame[0], os.path.join(pred_root, f"pred_seq{idx}.gif"))

    generated_frame = generated_frame.reshape(630, -1)
    return generated_frame


def test(args):
    os.makedirs(args.pred_root, exist_ok=True)

    # Load model
    model = VAE_Model(args).to(args.device)
    if not args.model_path:
        raise ValueError("Please specify the model path")
    model.load_state_dict(torch.load(args.model_path, weights_only=True))

    # Load data
    test_loader = get_dataloader(
        root=args.dataset_root,
        frame_H=args.frame_H,
        frame_W=args.frame_W,
        mode="test",
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
            pred_seq = test_step(model, img, label, args.device, idx, args.pred_root)
            pred_seq_list.append(pred_seq)

    save_submission(pred_seq_list, args.pred_root)


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
