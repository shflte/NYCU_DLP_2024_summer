import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser

from model.SCCNet import SCCNet
from utils import show_accuracy, show_learning_curve, parse_args
from Dataloader import MIBCI2aDataset


def main():
    # parse args
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=300, help='the number of epochs')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('-o', '--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='batch size')

    args = parser.parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    optimizer = args.optimizer
    batch_size = args.batch_size

    # dataset
    train_dataset = MIBCI2aDataset("train")
    test_dataset = MIBCI2aDataset("test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model
    features, _ = next(iter(train_loader))
    timeSample = features.shape[3]
    C = features.shape[2]
    model = SCCNet(numClasses=4, timeSample=timeSample, Nu=22, C=C, Nc=44, Nt=1, dropoutRate=0.5)
    model = model.cuda()

    # optimizer
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # training
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for epoch in tqdm(range(epochs)):
        model.train()
        correct = 0
        total = 0
        for features, labels in train_loader:
            features, labels = features.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

        train_loss.append(loss.item())
        train_acc.append(correct / total)

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

        test_loss.append(loss.item())
        test_acc.append(correct / total)

    # save model
    torch.save(model.state_dict(), f"model/trained/model.pth")

    # show result
    show_accuracy(train_acc, test_acc)
    show_learning_curve(train_loss, test_loss)


if __name__ == '__main__':
    main()
