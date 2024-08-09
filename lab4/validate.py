import torch
from utils import Generate_PSNR


def validate_step(model, images, labels, device):
    B, T, C, H, W = images.shape
    images = images.view(T, B, C, H, W)
    labels = labels.view(T, B, C, H, W)
    total_psnr = 0.0

    last_pred = None

    for t in range(1, T):
        img = images[t - 1] if last_pred is None else last_pred

        # encode
        img_features = model.frame_transformation(img)
        label_features = model.label_transformation(labels[t])

        # gaussian predictor
        z = torch.randn(B, 12, H, W).to(device)
        output = model.Decoder_Fusion(img_features, label_features, z)

        # generate
        prediction = model.Generator(output)
        last_pred = prediction

        # PSNR
        psnr = Generate_PSNR(prediction, images[t])
        total_psnr += psnr.mean().item()

    avg_psnr = total_psnr / (T - 1)
    return avg_psnr


def validate(model, val_loader, device):
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            total_psnr += validate_step(model, images, labels, device)

    avg_psnr = total_psnr / len(val_loader)
    print(f"Validation PSNR: {avg_psnr:.2f} dB")
    return avg_psnr
