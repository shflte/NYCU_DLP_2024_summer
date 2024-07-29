import torch
import argparse
from torch.utils.data import DataLoader
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score
from oxford_pet import load_dataset


def inference(args):
    # dataset
    test_dataset = load_dataset(data_path=args.data_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # model
    if args.model_type == "unet":
        model = UNet()
    elif args.model_type == "resnet34_unet":
        model = ResNet34_UNet()
    else:
        raise ValueError("Model not supported")
    model = model.cuda()
    # load the model weights
    model.load_state_dict(torch.load(args.model, weights_only=True))

    # test
    model.eval()
    with torch.no_grad():
        epoch_test_acc = 0
        test_batches = 0
        for batch in test_loader:
            image = batch['image'].cuda()
            mask = batch['mask'].cuda()
            output = model(image)

            # convert output to binary mask
            pred = (output > 0.5).float()
            pred = pred.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            acc = dice_score(pred, mask)
            epoch_test_acc += acc
            test_batches += 1

    print(f"Test Dice Score: {epoch_test_acc / test_batches}")


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model_path', '-mp', default='../saved_models/unet.pth', help='path to the stored model weight')
    parser.add_argument('--model_type', '-t', default='unet', help='model type')
    parser.add_argument('--data_path', '-p', type=str, default='../dataset/oxford-iiit-pet', help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    inference(args)
