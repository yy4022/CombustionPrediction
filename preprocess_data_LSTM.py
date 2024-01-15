import numpy as np
import torch
from torch.utils.data import DataLoader

from CAE.predict import CAE_predict
from methods_preprocess import MyDataset

# PART 1: define the parameters
# 1.1. choose the device
torch.cuda.set_device(0)

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

# 1.2. define the parameters for training the model
n_steps_in = 10
n_steps_out = 10
batch_size = 100
dataset_num = 2
model_title = 'PIV-x'
file_title = 'PIV_x'

# PART 2: load the encoder model
encoder = torch.load(f'model/encoder_{model_title}.pt')

# PART 3: load all the datasets
for i in range(dataset_num):
    # load the dataset
    initial_data = np.load(f'IA_{file_title}_{i+1}.npy')

    # create the corresponding dataloader
    initial_dataset = np.expand_dims(initial_data, axis=1)
    initial_dataset = MyDataset(initial_dataset)
    initial_dataloader = DataLoader(dataset=initial_dataset, batch_size=batch_size, shuffle=False)

    # encoder this dataset
    encoded_dataset = CAE_predict(CAE_model=encoder, device=device, dataloader_in=initial_dataloader)


