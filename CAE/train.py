import numpy as np
import torch
from torch import nn


def train_epoch(encoder: nn.Module, decoder: nn.Module, device: torch.device, dataloader_in, dataloader_out,
                loss_fn, optimizer):

    # set the train mode for encoder and decoder
    encoder.train()
    decoder.train()

    train_epoch_loss = []

    for image_batch_in, image_batch_out in zip(dataloader_in, dataloader_out):

        # move tensor to the proper device
        image_batch_in = image_batch_in.to(device)
        image_batch_out = image_batch_out.to(device)

        # 1. pass the input images to the encoder model
        encoded_data = encoder(image_batch_in)
        # 2. pass the encoded data to the decoder model
        predicted_data = encoder(encoded_data)

        # compute the prediction loss
        loss = loss_fn(predicted_data, image_batch_out)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_epoch_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_epoch_loss)
