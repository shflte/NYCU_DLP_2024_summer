# script for drawing figures, and more if needed
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def parse_args():
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

    return epochs, learning_rate, optimizer, batch_size


def show_accuracy(train_acc, test_acc):
    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, 'b', label='Training accuracy')
    plt.plot(epochs, test_acc, 'r', label='Test accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def show_learning_curve(train_loss, test_loss):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, test_loss, 'r', label='Test loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
