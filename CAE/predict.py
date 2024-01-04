import torch
from torch import nn

"""
This file is used for generating the predicted result by encoder/decoder model.
"""

def CAE_predict(CAE_model: nn.Module, device: torch.device, dataloader_in):

    # set evaluation mode for the given CAE model (encoder/decoder)
    CAE_model.eval()

    with torch.no_grad():

        predicted_output = []

        for image_batch_in in dataloader_in:

            # move the tensor to device
            image_batch_in = image_batch_in.to(device)

            # pass the input data to the CAE model (encoder/decoder)
            predicted_data = CAE_model(image_batch_in)

            predicted_output.append(predicted_data)

        predicted_output = torch.cat(predicted_output)

    return predicted_output
