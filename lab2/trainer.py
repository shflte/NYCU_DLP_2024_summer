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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MIBCI2aDataset("test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model
    model = SCCNet().cuda()
    if fine_tune:
        model.load_state_dict(torch.load("model/trained/loso_model.pth"))

    # optimizer
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # training
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        model.train()
        correct = 0
        total = 0
        for features, labels in train_loader:
            features, labels = features.cuda(), labels.cuda()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()

            _, predicted = torch.max(outputs, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

        optimizer.step()

        train_loss.append(loss.item())
        train_acc.append(correct / total)

        with torch.no_grad():
            val_correct = 0
            val_total = 0
            if epoch % 100 == 0:
                for features, labels in test_loader:
                    features, labels = features.cuda(), labels.cuda()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_size
                    val_correct += (predicted == labels).sum().item()
                val_loss.append(loss.item())
                val_acc.append(val_correct / val_total)
                print("val_acc: ", val_correct / val_total)
                print("val_loss: ", loss.item())

    # save model
    model_name = mode + "_model.pth"
    torch.save(model.state_dict(), f"model/trained/{model_name}")

    # show result
    show_accuracy(train_acc, "train")
    show_learning_curve(train_loss, "train")

    show_accuracy(val_acc, "val")
    show_learning_curve(val_loss, "val")


if __name__ == '__main__':
    # parse args
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=500, help='the number of epochs')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('-o', '--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('-b', '--batch-size', type=int, default=512, help='batch size')
    parser.add_argument('-m', '--mode', type=str, default='sd', help='sd, loso, loso + ft')

    args = parser.parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    optimizer = args.optimizer
    batch_size = args.batch_size
    mode = args.mode

    train(epochs, learning_rate, optimizer, batch_size, mode, fine_tune=(mode == "loso_ft"))
