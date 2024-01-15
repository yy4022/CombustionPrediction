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

def predict(encoder: nn.Module, decoder: nn.Module, device: torch.device,
            dataloader_in):

    # set evaluation mode for encoder and decoder model
    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        predicted_output = []

        for image_batch_in in dataloader_in:

            # move tensor to device
            image_batch_in = image_batch_in.to(device)

            # 1. pass the input data to the encoder model
            encoded_data = encoder(image_batch_in)
            # 2. pass the encoded data to the decoder model
            predicted_data = decoder(encoded_data)

            predicted_output.append(predicted_data)

        predicted_output = torch.cat(predicted_output)

    return predicted_output
