import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser

from model.SCCNet import SCCNet
from utils import show_accuracy, show_learning_curve
from Dataloader import MIBCI2aDataset


def train(epochs, learning_rate, optimizer, batch_size, mode, fine_tune=False):
    # dataset
    train_dataset = MIBCI2aDataset("train") if not fine_tune else MIBCI2aDataset("finetune")
    test_dataset = MIBCI2aDataset("test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model
    features, _ = next(iter(train_loader))
    _, _, C, timeSample = features.shape
    model = SCCNet(numClasses=4, timeSample=timeSample, Nu=22, C=C, Nc=44, Nt=1, dropoutRate=0.5)
    model = model.cuda()
    if fine_tune:
        model.load_state_dict(torch.load("model/trained/loso_model.pth"))

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
    model_name = mode + "_model.pth"
    torch.save(model.state_dict(), f"model/trained/{model_name}")

    # show result
    show_accuracy(train_acc, "train")
    show_learning_curve(train_loss, "train")

    show_accuracy(test_acc, "test")
    show_learning_curve(test_loss, "test")


if __name__ == '__main__':
    # parse args
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=300, help='the number of epochs')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('-o', '--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('-m', '--mode', type=str, default='sd', help='sd, loso, loso + ft')

    args = parser.parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    optimizer = args.optimizer
    batch_size = args.batch_size
    mode = args.mode

    train(epochs, learning_rate, optimizer, batch_size, mode)
    if mode == 'loso_ft':
        train(epochs, learning_rate, optimizer, batch_size, mode, fine_tune=True)
