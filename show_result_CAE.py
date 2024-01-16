import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from CAE.neural_net2 import Encoder, Decoder
from CAE.predict import predict
from CAE.validate import validate_epoch
from methods_show import show_difference, show_comparison

"""
This file is used for showing the results via the trained CAE model.
"""

# PART 1: define the parameters
# 1.1. choose the device
torch.cuda.set_device(0)

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

# 1.2. define the parameters and filenames
batch_size = 50
dataset_num = 2
img_scale = 373
specified_data = 200
model_title = 'PLIF'

testing_data_files = ['data/data4models/testing_PLIF_data1.npy',
                      'data/data4models/testing_PLIF_data2.npy']

testing_dataset_files = ['data/data4models/testing_PLIF_dataset1.pkl',
                         'data/data4models/testing_PLIF_dataset2.pkl']

# PART 2: load the existing model for showing results
# 1. define the Encoder and Decoder model
encoder = Encoder(img_scale)
encoder = torch.load(f'model/encoder_{model_title}.pt')

decoder = Decoder(img_scale)
decoder = torch.load(f'model/decoder_{model_title}.pt')

# 2. define the loss function
loss_fn = nn.MSELoss()

# PART 3: create the dataloader for showing
testing_dataloader_list = []

for i in range(dataset_num):
    # 1. load the dataset
    with open(testing_dataset_files[i], 'rb') as file:
        testing_dataset = pickle.load(file)

    # 2. create the corresponding dataloader
    testing_dataloader = DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=False)

    # 3. append to the dataloader list
    testing_dataloader_list.append(testing_dataloader)

# PART 4: calculate the loss for testing dataset
testing_loss = 0.0

for i in range(dataset_num):

    testing_loss_i = validate_epoch(encoder=encoder, decoder=decoder, device=device,
                                    dataloader_in=testing_dataloader_list[i],
                                    dataloader_out=testing_dataloader_list[i],
                                    loss_fn=loss_fn)

    testing_loss = testing_loss + testing_loss_i

testing_loss = testing_loss / dataset_num
print(f'The MSE loss for the testing dataset is {testing_loss}.')

# PART 5: show the difference and comparison of the specified data
for i in range(dataset_num):

    predicted_data_i = predict(encoder=encoder, decoder=decoder, device=device,
                               dataloader_in=testing_dataloader_list[i])
    predicted_data_i = predicted_data_i.cpu()
    predicted_data_i = predicted_data_i.numpy()
    predicted_data_i = np.squeeze(predicted_data_i)

    original_data_i = np.load(testing_data_files[i])
    show_comparison(original_data=original_data_i[specified_data], prediction_data=predicted_data_i[specified_data],
                    original_title=f'Original_{model_title}{i}', prediction_title=f'Prediction_{model_title}{i}',
                    vmin=min(np.min(predicted_data_i), np.min(original_data_i)),
                    vmax=max(np.max(predicted_data_i), np.max(original_data_i)))

    difference_i = predicted_data_i - original_data_i
    show_difference(image_data=difference_i[specified_data], title=f'Difference_{model_title}{i}',
                    vmin=np.min(difference_i), vmax=np.max(difference_i))

