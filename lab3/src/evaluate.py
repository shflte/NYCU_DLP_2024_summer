import torch
from utils import dice_score


def evaluate(net, data, criterion, device):
    # validation
    net.eval()
    epoch_val_loss = 0
    epoch_val_acc = 0
    val_batches = 0
    with torch.no_grad():
        for batch in data:
            image = batch["image"].cuda()
            mask = batch["mask"].cuda()
            output = net(image)
            loss = criterion(output, mask)

            # convert output to binary mask
            pred = (output > 0.5).float()
            pred = pred.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            acc = dice_score(pred, mask)
            epoch_val_acc += acc
            epoch_val_loss += loss.item()
            val_batches += 1

    return epoch_val_loss / val_batches, epoch_val_acc / val_batches
