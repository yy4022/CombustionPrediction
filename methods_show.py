from typing import Dict, List

import matplotlib
import numpy as np
import torch

from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from CAE.predict import CAE_predict
from methods_preprocess import MyDataset


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


def show_normalized_image(image_data: np.ndarray, vmin: float, vmax: float, title: str):

    plt.figure(figsize=(16, 12))

    plt.title(title, fontsize=20)
    plt.imshow(image_data, cmap='turbo', origin='lower', interpolation='bicubic',
               extent=(-18, 18, 0, 36), vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    plt.show()

def decode_results(decoder_PLIF: nn.Module, decoder_PIV_x: nn.Module, decoder_PIV_y: nn.Module,
                   decoder_PIV_z: nn.Module, predicted_output: np.ndarray,
                   features: int, batch_size: int, device: torch.device):

    # split the predicted output
    predicted_PLIF = predicted_output[:, 0:features]
    predicted_PIV_x = predicted_output[:, features:2*features]
    predicted_PIV_y = predicted_output[:, 2*features:3*features]
    predicted_PIV_z = predicted_output[:, 3*features:4*features]

    # create the dataloaders
    predicted_PLIF_dataset = MyDataset(predicted_PLIF)
    predicted_PIV_x_dataset = MyDataset(predicted_PIV_x)
    predicted_PIV_y_dataset = MyDataset(predicted_PIV_y)
    predicted_PIV_z_dataset = MyDataset(predicted_PIV_z)

    predicted_PLIF_dataloader = DataLoader(dataset=predicted_PLIF_dataset, batch_size=batch_size, shuffle=False)
    predicted_PIV_x_dataloader = DataLoader(dataset=predicted_PIV_x_dataset, batch_size=batch_size, shuffle=False)
    predicted_PIV_y_dataloader = DataLoader(dataset=predicted_PIV_y_dataset, batch_size=batch_size, shuffle=False)
    predicted_PIV_z_dataloader = DataLoader(dataset=predicted_PIV_z_dataset, batch_size=batch_size, shuffle=False)

    # decode the predicted data
    decoded_PLIF = CAE_predict(CAE_model=decoder_PLIF, device=device, dataloader_in=predicted_PLIF_dataloader)
    decoded_PIV_x = CAE_predict(CAE_model=decoder_PIV_x, device=device, dataloader_in=predicted_PIV_x_dataloader)
    decoded_PIV_y = CAE_predict(CAE_model=decoder_PIV_y, device=device, dataloader_in=predicted_PIV_y_dataloader)
    decoded_PIV_z = CAE_predict(CAE_model=decoder_PIV_z, device=device, dataloader_in=predicted_PIV_z_dataloader)

    decoded_PLIF = np.squeeze(decoded_PLIF.cpu().numpy())
    decoded_PIV_x = np.squeeze(decoded_PIV_x.cpu().numpy())
    decoded_PIV_y = np.squeeze(decoded_PIV_y.cpu().numpy())
    decoded_PIV_z = np.squeeze(decoded_PIV_z.cpu().numpy())

    return decoded_PLIF, decoded_PIV_x, decoded_PIV_y, decoded_PIV_z

def show_comparison(original_data: np.ndarray, prediction_data: np.ndarray,
                    original_title: str, prediction_title: str,
                    vmin: float, vmax: float):

    show_normalized_image(image_data=original_data, title=original_title, vmin=vmin, vmax=vmax)

    show_normalized_image(image_data=prediction_data, title=prediction_title, vmin=vmin, vmax=vmax)

def show_difference(image_data: np.ndarray, title: str, vmin: float, vmax: float):

    show_normalized_image(image_data=image_data, title=title, vmin=vmin, vmax=vmax)

def show_loss(loss: Dict[str, np.ndarray], filename: str):

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
