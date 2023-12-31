import numpy as np
import torch
from torch import nn


def validate_epoch(encoder: nn.Module, decoder: nn.Module, device: torch.device,
                   dataloader_in, dataloader_out, loss_fn):

    # set evaluation mode for encoder and decoder model
    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        validate_epoch_loss = []

        for image_batch_in, image_batch_out in zip(dataloader_in, dataloader_out):

            # move tensor to device
            image_batch_in = image_batch_in.to(device)
            image_batch_out = image_batch_out.to(device)

            # 1. pass the input data to the encoder model
            encoded_data = encoder(image_batch_in)
            # 2. pass the encoded data to the decoder model
            predicted_data = decoder(encoded_data)

            # compute the prediction loss
            loss = loss_fn(predicted_data, image_batch_out)

            validate_epoch_loss.append(loss.detach().cpu().numpy())

    return np.mean(validate_epoch_loss)
