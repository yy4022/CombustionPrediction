import numpy as np
import torch
from torch import nn


def predict_lstm(lstm_model: nn.Module, device: torch.device, dataloader_in):

    # set the evaluation mode for the lstm model
    lstm_model.eval()

    with torch.no_grad():

        predicted_output = []

        for image_batch_in in dataloader_in:

            # move tensor to device
            image_batch_in = image_batch_in.to(device)

            # pass the image to the lstm model
            predicted_data = lstm_model(image_batch_in)

            # append the predicted output to the lists
            predicted_output.append(predicted_data.cpu())

        # create a single tensor with all the values in the list
        predicted_output = torch.cat(predicted_output)

    return predicted_output
