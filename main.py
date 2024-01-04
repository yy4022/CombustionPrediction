import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from CAE.neural_net import Encoder, Decoder

# PART 1: define the parameters
# 1.1. choose the device
torch.cuda.set_device(0)

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

# 1.2. define the parameters for training the model
batch_size = 200
EPOCHS = 1000
lr = 0.0001
if_existing = False  # a flag recording if there is a set of existing model
img_scale = 373
latent_features = 100
dataset_num = 2
model_title = 'PLIF'

# 1.3. define the file list
training_files = ['data/data4models/training_PLIF_dataset1.pkl',
                  'data/data4models/training_PLIF_dataset2.pkl']

validation_files = ['data/data4models/validation_PLIF_dataset1.pkl',
                    'data/data4models/validation_PLIF_dataset2.pkl']

# PART 2: create the dataloader for training
training_dataloader_list = []
validation_dataloader_list = []

for i in range(dataset_num):
    # 1. load the dataset
    with open(training_files[i], 'rb') as file:
        training_dataset = pickle.load(file)

    with open(validation_files[i], 'rb') as file:
        validation_dataset = pickle.load(file)

    # 2. create the corresponding dataloader
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=False)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

    # 3. append to the dataloader list
    training_dataloader_list.append(training_dataloader)
    validation_dataloader_list.append(validation_dataloader)

# PART 3: preparation before training the model
# 1. define the Encoder model
encoder = Encoder(img_scale)

# check if there is an existing model
if os.path.exists(f'./model/encoder_{model_title}.pt'):
    encoder = torch.load(f'./model/encoder_{model_title}.pt')

    # set if_existing flag
    if_existing = True
    print("Load the existing Encoder model, then continue training.")
else:
    print("No existing Encoder model, so create a new one.")

encoder = encoder.to(device)
input_shape = (1, img_scale, img_scale)
summary(encoder, input_shape)

# 2. define the Decoder model
decoder = Decoder(img_scale)

# check if there is an existing model
if if_existing:
    decoder = torch.load(f'./model/decoder_{model_title}.pt')
    print("Load the existing Decoder model, then continue training.")
else:
    print("No existing Decoder model, so create a new one.")

decoder = decoder.to(device)
input_shape = (latent_features,)
summary(decoder, input_shape)

# 3. create two numpy arrays for recording the loss,
#   and set the best (validation) loss for updating the mode
train_loss_records = []
validation_loss_records = []

train_loss_records = np.array(train_loss_records)
validation_loss_records = np.array(validation_loss_records)

best_loss = 10.0


