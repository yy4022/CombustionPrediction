import os

import torch
from torchsummary import summary

from CAE.neural_net import Encoder, Decoder

# PART 1: define the parameters
# 1.1. choose the device
torch.cuda.set_device(0)

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

# PART 2: preparation before training the model
# 1. define the Encoder model
img_scale = 373
encoder = Encoder(img_scale)

encoder = encoder.to(device)
input_shape = (1, img_scale, img_scale)
summary(encoder, input_shape)

# 2. define the Decoder model
img_scale = 373
decoder = Decoder(img_scale)

decoder = decoder.to(device)
input_shape = (100,)
summary(decoder, input_shape)

