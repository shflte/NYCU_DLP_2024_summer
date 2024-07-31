import numpy as np
import matplotlib.pyplot as plt


def dice_score(pred_mask, gt_mask):
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    return 2 * intersection / union


def show_accuracy(acc, fileName):
    plt.figure()
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, "b", label="Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.savefig(f"{fileName}_accuracy.png")


def show_learning_curve(loss, fileName):
    plt.figure()
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, "b", label="Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(f"{fileName}_loss.png")


def show_image(image, pred_mask, gt_mask, name):
    plt.figure()

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask)
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(gt_mask)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.savefig(f"../predictions/{name}.png")
