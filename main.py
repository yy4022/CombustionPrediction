import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from CAE.neural_net import Encoder, Decoder
from CAE.train import train_epoch
from CAE.validate import validate_epoch
from methods_show import show_loss

# PART 1: define the parameters
# 1.1. choose the device
torch.cuda.set_device(0)

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

# 1.2. define the parameters for training the model
batch_size = 200
EPOCHS = 10
lr = 0.0001
if_existing = False  # a flag recording if there is a set of existing model
img_scale = 71
latent_features = 100
dataset_num = 2
model_title = 'PIV-x'

# 1.3. define the file list
training_files = ['data/data4models/training_PIV_x_dataset1.pkl',
                  'data/data4models/training_PIV_x_dataset2.pkl']

validation_files = ['data/data4models/validation_PIV_x_dataset1.pkl',
                    'data/data4models/validation_PIV_x_dataset2.pkl']

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

if if_existing:
    train_loss_records = \
        np.append(train_loss_records, np.load(f'./result/train_loss_records_{model_title}.npy'))

    validation_loss_records = \
        np.append(validation_loss_records, np.load(f'./result/validation_loss_records_{model_title}.npy'))

    best_loss = validation_loss_records.min()
    print(f"Load the existing loss records, and current best loss is {best_loss}.")

else:
    print("No existing loss records, start recording from the beginning.")

# 4. define the loss function and the optimizer
loss_fn = nn.MSELoss()

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

parameters_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optimizer = torch.optim.Adam(parameters_to_optimize, lr=lr, weight_decay=1e-05)

# PART 4: the training part
for epoch in range(EPOCHS):

    train_loss_list = []
    validation_loss_list = []

    for i in range(dataset_num):
        train_loss_i = train_epoch(encoder=encoder, decoder=decoder, device=device,
                                   dataloader_in=training_dataloader_list[i],
                                   dataloader_out=training_dataloader_list[i],
                                   loss_fn=loss_fn, optimizer=optimizer)

        validation_loss_i = validate_epoch(encoder=encoder, decoder=decoder, device=device,
                                           dataloader_in=validation_dataloader_list[i],
                                           dataloader_out=validation_dataloader_list[i],
                                           loss_fn=loss_fn)

        train_loss_list.append(train_loss_i)
        validation_loss_list.append(validation_loss_i)

    train_loss = np.mean(train_loss_list)
    validation_loss = np.mean(validation_loss_list)

    print(
        '\n EPOCH {}/{} \t train loss {} \t validate loss {}'.format(epoch + 1, EPOCHS, train_loss,
                                                                     validation_loss))

    train_loss_records = np.append(train_loss_records, train_loss)
    validation_loss_records = np.append(validation_loss_records, validation_loss)

    if validation_loss < best_loss:
        best_loss = validation_loss
        torch.save(encoder, f'./model/encoder_{model_title}.pt')
        torch.save(decoder, f'./model/decoder_{model_title}.pt')

# save loss records of training and validation process
np.save(f'./result/train_loss_records_{model_title}.npy', train_loss_records)
np.save(f'./result/validation_loss_records_{model_title}.npy', validation_loss_records)

loss_records = {
    'train_loss_records': train_loss_records,
    'validation_loss_records': validation_loss_records
}

# PART 5: show the results
# 5.1. show the loss records of the whole training process
show_loss(loss_records, f"CAE_loss_{model_title}.png")
