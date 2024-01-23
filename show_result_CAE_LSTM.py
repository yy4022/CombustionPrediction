import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from CAE.predict import CAE_predict
from LSTM.predict import predict_lstm
from LSTM.validate import validate_epoch
from methods_preprocess import MyDataset
from methods_show import decode_results, show_comparison, show_difference

# from methods_show import show_lstm_comparison, show_lstm_difference

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
batch_size = 10
specified_seq = 0
specified_image = 4
features = 100

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
# use the lstm model to get result
predicted_lstm_output = predict_lstm(lstm_model=lstm_model, device=device, dataloader_in=testing_in_dataloader)
predicted_lstm_output = predicted_lstm_output.cpu().numpy()

# use the decoder models to get results
lstm_cae_PLIF, lstm_cae_PIV_x, lstm_cae_PIV_y, lstm_cae_PIV_z = \
    decode_results(decoder_PLIF, decoder_PIV_x, decoder_PIV_y, decoder_PIV_z,
                   predicted_lstm_output[specified_seq, :, :],
                   features=features, batch_size=batch_size, device=device)

cae_PLIF, cae_PIV_x, cae_PIV_y, cae_PIV_z = \
    decode_results(decoder_PLIF, decoder_PIV_x, decoder_PIV_y, decoder_PIV_z,
                   testing_out_data[specified_seq, :, :],
                   features=features, batch_size=batch_size, device=device)

difference_PLIF = lstm_cae_PLIF - cae_PLIF
difference_PIV_x = lstm_cae_PIV_x - cae_PIV_x
difference_PIV_y = lstm_cae_PIV_y - cae_PIV_y
difference_PIV_z = lstm_cae_PIV_z - cae_PIV_z

# visualise the comparison
show_comparison(original_data=cae_PLIF[specified_image, :, :], prediction_data=lstm_cae_PLIF[specified_image, :, :],
                original_title='lstm_cae_PLIF', prediction_title='cae_PLIF', vmin=0.0, vmax=1.0)

show_comparison(original_data=cae_PIV_x[specified_image, :, :], prediction_data=lstm_cae_PIV_x[specified_image, :, :],
                original_title='lstm_cae_PIV_x', prediction_title='cae_PIV_x', vmin=0.0, vmax=1.0)
show_comparison(original_data=cae_PIV_y[specified_image, :, :], prediction_data=lstm_cae_PIV_y[specified_image, :, :],
                original_title='lstm_cae_PIV_y', prediction_title='cae_PIV_y', vmin=0.0, vmax=1.0)
show_comparison(original_data=cae_PIV_z[specified_image, :, :], prediction_data=lstm_cae_PIV_z[specified_image, :, :],
                original_title='lstm_cae_PIV_z', prediction_title='cae_PIV_z', vmin=0.0, vmax=1.0)

show_difference(image_data=difference_PLIF[specified_image, :, :], title='difference_PLIF', vmin=0.0, vmax=1.0)
show_difference(image_data=difference_PIV_x[specified_image, :, :], title='difference_PIV_x', vmin=0.0, vmax=1.0)
show_difference(image_data=difference_PIV_y[specified_image, :, :], title='difference_PIV_y', vmin=0.0, vmax=1.0)
show_difference(image_data=difference_PIV_z[specified_image, :, :], title='difference_PIV_z', vmin=0.0, vmax=1.0)

"""
SECTION 2: show the predicted results influenced by the LSTM and CAE models simultaneously.
NOTE: you can comment this section if you do not need this function.
"""

"""
SECTION 3: show the predicted results by iterative prediction.
"""
