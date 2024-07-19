import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.SCCNet import SCCNet
from utils import parse_args
from Dataloader import MIBCI2aDataset


def main():
    _, _, _, batch_size = parse_args()

    # dataset
    test_dataset = MIBCI2aDataset("test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model
    features, _ = next(iter(test_loader))
    timeSample = features.shape[3]
    C = features.shape[2]
    model = SCCNet(numClasses=4, timeSample=timeSample, Nu=22, C=C, Nc=44, Nt=1, dropoutRate=0.5)
    model = model.cuda()
    model.load_state_dict(torch.load("model/trained/model.pth"))

    # loss function
    criterion = nn.CrossEntropyLoss()

    # testing
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.cuda(), labels.cuda()
            outputs = model(features)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {correct / total * 100:.2f}%")


if __name__ == '__main__':
    main()
