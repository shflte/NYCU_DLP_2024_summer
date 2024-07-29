import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import show_accuracy, show_learning_curve, dice_score
from evaluate import evaluate
from oxford_pet import load_dataset


def train(args):
    # dataset
    train_dataset = load_dataset(data_path=args.data_path, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = load_dataset(data_path=args.data_path, mode="valid")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # model
    if args.model_type == "unet":
        model = UNet()
    elif args.model_type == "resnet34_unet":
        model = ResNet34_UNet()
    else:
        raise ValueError("Model not supported")
    model = model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # loss function
    criterion = torch.nn.BCELoss()

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for epoch in tqdm(range(args.epochs)):
        # train
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        train_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            image = batch['image'].cuda()
            mask = batch['mask'].cuda()
            output = model(image)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()

            # convert output to binary mask
            pred = (output > 0.5).float()
            pred = pred.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            acc = dice_score(pred, mask)
            epoch_train_acc += acc
            epoch_train_loss += loss.item()
            train_batches += 1

        train_loss.append(epoch_train_loss / train_batches)
        train_acc.append(epoch_train_acc / train_batches)

        # validation
        val_loss_epoch, val_acc_epoch = evaluate(model, val_loader, criterion, device="cuda")
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)

    # save model
    torch.save(model.state_dict(), f"{args.model_path}/{args.model}.pth")

    # show results
    show_accuracy(train_acc, args.model + "_train")
    show_learning_curve(train_loss, args.model + "_train")

    show_accuracy(val_acc, args.model + "_val")
    show_learning_curve(val_loss, args.model + "_val")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model_path', '-m', type=str, default='../saved_models', help='path to save the model')
    parser.add_argument('--model_type', '-t', type=str, default='unet', help='model to use (unet, resnet34_unet)')
    parser.add_argument('--data_path', '-p', default='../dataset/oxford-iiit-pet', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
