import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from LSTM.validate import validate_epoch
from methods_preprocess import MyDataset

"""
This file is used for showing the results via the trained CAE and LSTM model.

NOTE: we need to show two results:
    1. firstly, we should see the predicted results only influenced by the LSTM model, 
       which means compared with the predicted results of CAE model, instead of the original data.
    2. then, we should see the predicted results influenced by the LSTM and CAE models simultaneously,
       which means compared with the original data.
    3. finally, you can evaluate both influences by iterative prediction process.
"""

"""
SECTION 1: show the predicted results only influenced by the LSTM model.
NOTE: you can comment this section if you do not need this function.
"""

# PART 1: define the parameters
# 1.1. choose the device
torch.cuda.set_device(0)

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

# 1.2. define the parameters and filenames
batch_size = 100
specified_data = 200

testing_in_file = 'data/data4LSTM/testing_sequence_in.npy'
testing_out_file = 'data/data4LSTM/testing_sequence_out.npy'

# PART 2: load the existing model for showing results
# 1. define the Decoder and LSTM models
decoder_PLIF = torch.load('model/decoder_PLIF.pt')
decoder_PIV_x = torch.load('model/decoder_PIV-x.pt')
decoder_PIV_y = torch.load('model/decoder_PIV-y.pt')
decoder_PIV_z = torch.load('model/decoder_PIV-z.pt')

lstm_model = torch.load('model/LSTM.pt')

# 2. define the loss function
loss_fn = nn.MSELoss()

# PART 3: create the dataloader for showing
# load the data
testing_in_data = np.load(testing_in_file)
testing_out_data = np.load(testing_out_file)

# create the dataset
testing_in_dataset = MyDataset(testing_in_data)
testing_out_dataset = MyDataset(testing_out_data)

# create the dataloaders
testing_in_dataloader = DataLoader(dataset=testing_in_dataset, batch_size=batch_size, shuffle=False)
testing_out_dataloader = DataLoader(dataset=testing_out_dataset, batch_size=batch_size, shuffle=False)

# PART 4: calculate the loss for testing dataset
testing_loss = validate_epoch(lstm_model=lstm_model, device=device, dataloader_in=testing_in_dataloader,
                              dataloader_out=testing_out_dataloader, loss_fn=loss_fn)

print(f'The MSE loss for the testing dataset is {testing_loss}.')

# PART 5: show the difference and comparison of the specified data


"""
SECTION 2: show the predicted results influenced by the LSTM and CAE models simultaneously.
NOTE: you can comment this section if you do not need this function.
"""

"""
SECTION 3: show the predicted results by iterative prediction.
"""
