from typing import Dict, List

import matplotlib
import numpy as np

from matplotlib import pyplot as plt


def show_image(image_data: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float, title: str):

    """
    Show the given image:
    :param image_data: a numpy array represents the value of the given image data
    :param xmin: the min value of x-axis for this image
    :param xmax: the max value of x-axis for this image
    :param ymin: the min value of y-axis for this image
    :param ymax: the max value of y-axis for this image
    :param title: the tile of this image
    """

    plt.figure(figsize=(16, 12))

    plt.title(title, fontsize=20)
    plt.imshow(image_data, cmap='turbo', origin='lower', interpolation='bicubic',
               extent=(xmin, xmax, ymin, ymax))
    cbar = plt.colorbar()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    plt.show()


def show_loss(loss: Dict[str, List[float]], filename: str):

    """
    Visualizes the training and validation loss over epochs.

    :param loss: A dictionary contains training and validation loss records.
    :param filename: A string contains the name for saving the plot.
    :return: None.
    """

    plt.figure(figsize=(10, 8))
    plt.semilogy(loss['train_loss_records'], label='Train')
    plt.semilogy(loss['validation_loss_records'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()

    plt.savefig(f"./result/{filename}")
    plt.show()