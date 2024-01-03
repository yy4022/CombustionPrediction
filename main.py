import os

import torch
from torchsummary import summary

from CAE.neural_net import Encoder

# PART 1: define the parameters
# 1.1. choose the device
torch.cuda.set_device(0)

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

# PART 2: preparation before training the model
# 1. define the FullyCNN model
img_scale = 373
fullyCNN = Encoder(img_scale)

fullyCNN = fullyCNN.to(device)
input_shape = (1, img_scale, img_scale)
summary(fullyCNN, input_shape)
