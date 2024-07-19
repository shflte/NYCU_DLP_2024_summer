import matplotlib.pyplot as plt


def show_accuracy(acc, fileName):
    plt.figure()
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig(f'{fileName}_accuracy.png')


def show_learning_curve(loss, fileName):
    plt.figure()
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'b', label='Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(f'{fileName}_loss.png')
