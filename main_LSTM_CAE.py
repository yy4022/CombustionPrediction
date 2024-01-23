import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from LSTM.neural_net import LSTM
from LSTM.train import train_epoch
from LSTM.validate import validate_epoch
from methods_preprocess import MyDataset
from methods_show import show_loss

# PART 1: define the parameters
# 1.1. choose the device
torch.cuda.set_device(0)

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

# 1.2. define the parameters for training the model
EPOCHS = 100
batch_size = 100
lr = 0.0001
if_existing = False  # a flag recording if there is a set of existing model

# 1.3. define the filenames
training_in_file = 'data/data4LSTM/training_sequence_in.npy'
training_out_file = 'data/data4LSTM/training_sequence_out.npy'

validation_in_file = 'data/data4LSTM/validation_sequence_in.npy'
validation_out_file = 'data/data4LSTM/validation_sequence_out.npy'

# PART 2: create the training and validation dataloaders
# 1. load the files
training_in_data = np.load(training_in_file)
training_out_data = np.load(training_out_file)

validation_in_data = np.load(validation_in_file)
validation_out_data = np.load(validation_out_file)

# 2. create the datasets
training_in_dataset = MyDataset(training_in_data)
training_out_dataset = MyDataset(training_out_data)

validation_in_dataset = MyDataset(validation_in_data)
validation_out_dataset = MyDataset(validation_out_data)

# 3. generate the corresponding dataloaders
training_in_dataloader = DataLoader(dataset=training_in_dataset, batch_size=batch_size, shuffle=False)
training_out_dataloader = DataLoader(dataset=training_out_dataset, batch_size=batch_size, shuffle=False)

validation_in_dataloader = DataLoader(dataset=validation_in_dataset, batch_size=batch_size, shuffle=False)
validation_out_dataloader = DataLoader(dataset=validation_out_dataset, batch_size=batch_size, shuffle=False)

# PART 3: preparation for training the model
# 1. define the LSTM model
if os.path.exists('./model/LSTM.pt'):
    lstm_model = torch.load('./model/LSTM.pt')

    # set if_existing flag
    if_existing = True
    print("Load the existing LSTM model, then continue training.")
else:
    lstm_model = LSTM(input_size=400, hidden_size=128, num_layers=1, output_size=400,
                      batch_size=batch_size,
                      device=device)
    print("No existing LSTM model, but create a new LSTM model.")

lstm_model = lstm_model.to(device)
input_shape = (100, 10, 400)
summary(lstm_model, input_shape)

# 2. create two numpy arrays for recording the loss,
#    and set the best (validation) loss for updating the model
train_loss_records = []
validation_loss_records = []

train_loss_records = np.array(train_loss_records)
validation_loss_records = np.array(validation_loss_records)

best_loss = 10.0

if if_existing:
    train_loss_records = \
        np.append(train_loss_records, np.load(f'./result/train_loss_records_LSTM.npy'))

    validation_loss_records = \
        np.append(validation_loss_records, np.load(f'./result/validation_loss_records_LSTM.npy'))

    best_loss = validation_loss_records.min()
    print(f"Load the existing loss records, and current best loss is {best_loss}.")

else:
    print("No existing loss records, start recording from the beginning.")

# 3. define the loss function and the optimizer
loss_fn = nn.MSELoss()

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr, weight_decay=1e-05)

# PART 4: the looping process for training the model
for epoch in range(EPOCHS):

    train_loss = train_epoch(lstm_model=lstm_model, device=device, dataloader_in=training_in_dataloader,
                             dataloader_out=training_out_dataloader, loss_fn=loss_fn, optimizer=optimizer)

    validation_loss = validate_epoch(lstm_model=lstm_model, device=device, dataloader_in=training_in_dataloader,
                                     dataloader_out=training_out_dataloader, loss_fn=loss_fn)

    print(
        '\n EPOCH {}/{} \t train loss {} \t validate loss {}'.format(epoch + 1, EPOCHS, train_loss,
                                                                     validation_loss))

    train_loss_records = np.append(train_loss_records, train_loss)
    validation_loss_records = np.append(validation_loss_records, validation_loss)

    if validation_loss < best_loss:
        best_loss = validation_loss
        torch.save(lstm_model, f'./model/LSTM.pt')

# save loss records of training and validation process
np.save(f'./result/train_loss_records_LSTM.npy', train_loss_records)
np.save(f'./result/validation_loss_records_LSTM.npy', validation_loss_records)

loss_records = {
    'train_loss_records': train_loss_records,
    'validation_loss_records': validation_loss_records
}

# PART 5: show the results
# 5.1. show the loss records of the whole training process
show_loss(loss_records, f"LSTM_loss.png")
