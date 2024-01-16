import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from CAE.predict import CAE_predict
from methods_preprocess import MyDataset, split_dataset_overlap, split_dataset

"""
SECTION 1: encode, concatenate the initial datasets, then we can get the dataset of features in latent space.
NOTE: you can comment this SECTION after using.
"""

# # PART 1: define the parameters
# # 1.1. choose the device
# torch.cuda.set_device(0)
#
# print(torch.__version__)
# print(torch.cuda.is_available())
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(f"Selected device: {device}")
#
# # 1.2. define the parameters for training the model
# n_steps_in = 10
# n_steps_out = 10
# batch_size = 50
# dataset_num = 2
#
# # PART 2: load the encoder model
# encoder_PLIF = torch.load('model/encoder_PLIF.pt')
# encoder_PIV_x = torch.load('model/encoder_PIV-x.pt')
# encoder_PIV_y = torch.load('model/encoder_PIV-y.pt')
# encoder_PIV_z = torch.load('model/encoder_PIV-z.pt')
#
# def create_dataloader(my_data: np.ndarray, batch_size: int):
#
#     my_data = np.expand_dims(my_data, axis=1)
#     my_dataset = MyDataset(my_data)
#     my_dataloader = DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=False)
#
#     return my_dataloader
#
# # PART 3: load all the datasets
# for i in range(dataset_num):
#     # STEP 1: load the dataset
#     initial_data_PLIF = np.load(f'data/unshuffled_numpy_data/IA_PLIF_{i + 1}.npy')
#     initial_data_PIV_x = np.load(f'data/unshuffled_numpy_data/IA_PIV_x_{i + 1}.npy')
#     initial_data_PIV_y = np.load(f'data/unshuffled_numpy_data/IA_PIV_y_{i + 1}.npy')
#     initial_data_PIV_z = np.load(f'data/unshuffled_numpy_data/IA_PIV_z_{i + 1}.npy')
#
#     # STEP 2: create the corresponding dataloader
#     dataloader_PLIF = create_dataloader(my_data=initial_data_PLIF, batch_size=batch_size)
#     dataloader_PIV_x = create_dataloader(my_data=initial_data_PIV_x, batch_size=batch_size)
#     dataloader_PIV_y = create_dataloader(my_data=initial_data_PIV_y, batch_size=batch_size)
#     dataloader_PIV_z = create_dataloader(my_data=initial_data_PIV_z, batch_size=batch_size)
#
#     # STEP 3: encode these dataset
#     encoded_PLIF = CAE_predict(CAE_model=encoder_PLIF, device=device, dataloader_in=dataloader_PLIF)
#     encoded_PIV_x = CAE_predict(CAE_model=encoder_PIV_x, device=device, dataloader_in=dataloader_PIV_x)
#     encoded_PIV_y = CAE_predict(CAE_model=encoder_PIV_y, device=device, dataloader_in=dataloader_PIV_y)
#     encoded_PIV_z = CAE_predict(CAE_model=encoder_PIV_z, device=device, dataloader_in=dataloader_PIV_z)
#
#     # STEP 4: concatenate all these encoded data
#     encoded_features = torch.cat((encoded_PLIF, encoded_PIV_x, encoded_PIV_y, encoded_PIV_z), dim=1)
#
#     # STEP 5: save the encoded features in latent space
#     encoded_features = encoded_features.cpu().data.numpy()
#     np.save(f'data/data4LSTM/encoded_features_{i + 1}.npy', encoded_features)

"""
SECTION 2: generate the sequences datasets.
NOTE: you can comment this SECTION after using.
"""

# # PART 1: define the parameters
# # 1.1. choose the device
# torch.cuda.set_device(0)
#
# print(torch.__version__)
# print(torch.cuda.is_available())
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(f"Selected device: {device}")
#
# # 1.2. define the parameters for training the model
# n_steps_in = 10
# n_steps_out = 10
# batch_size = 50
# dataset_num = 2
#
# # PART 2: load all the datasets
# encoded_dataset = np.load('data/data4LSTM/encoded_features_1.npy')
#
# for i in range(1, dataset_num):
#     encoded_features = np.load(f'data/data4LSTM/encoded_features_{i + 1}.npy')
#     # load the encoded dataset
#     encoded_dataset = np.append(encoded_dataset, encoded_features, axis=0)
#
# # PART 3: split the encoded features into overlapping sequences (type: List)
# overlap_sequences_in, overlap_sequences_out = split_dataset_overlap(my_dataset=encoded_dataset,
#                                                                     n_step_in=n_steps_in, n_step_out=n_steps_out)
#
# print(len(overlap_sequences_in))
# print(len(overlap_sequences_out))
# print()
#
# # save the overlapping sequences (type: List)
# with open('data/data4LSTM/overlap_sequences_in.pkl', 'wb') as file:
#     pickle.dump(overlap_sequences_in, file)
#
# with open(f'data/data4LSTM/overlap_sequences_out.pkl', 'wb') as file:
#     pickle.dump(overlap_sequences_out, file)
#
# # PART 4: split the encoded features into normal sequences (type: List)
# sequences_in, sequences_out = split_dataset(my_dataset=encoded_dataset,
#                                             n_steps_in=n_steps_in, n_steps_out=n_steps_out)
#
# print(len(sequences_in))
# print(len(sequences_out))
#
# # save all the normal sequences (type: List)
# with open(f'data/data4LSTM/sequences_in.pkl', 'wb') as file:
#     pickle.dump(sequences_in, file)
#
# with open(f'data/data4LSTM/sequences_out.pkl', 'wb') as file:
#     pickle.dump(sequences_out, file)


"""
SECTION 3: preprocess the overlapped datasets of sequences into training, validation and testing datasets. 
"""

# PART 1: load all datasets
with open('data/data4LSTM/overlap_sequences_in.pkl', 'rb') as file:
    overlap_sequences_in = pickle.load(file)

with open('data/data4LSTM/overlap_sequences_out.pkl', 'rb') as file:
    overlap_sequences_out = pickle.load(file)

# PART 2: split the datasets
sequence_num = len(overlap_sequences_in)
split_points = [int(np.floor(sequence_num * 0.8)), int(np.floor(sequence_num * 0.9))]

sequence_in_split = np.split(overlap_sequences_in, split_points, axis=0)
sequence_out_split = np.split(overlap_sequences_out, split_points, axis=0)

training_sequence_in = sequence_in_split[0]
training_sequence_out = sequence_out_split[0]

validation_sequence_in = sequence_in_split[1]
validation_sequence_out = sequence_out_split[1]

testing_sequence_in = sequence_in_split[2]
testing_sequence_out = sequence_out_split[2]

print(len(overlap_sequences_in))
print(split_points)
print()

print(training_sequence_in.shape)
print(len(training_sequence_in))
print(len(validation_sequence_in))
print(len(testing_sequence_in))

# PART 3: save the datasets
np.save('data/data4LSTM/training_sequence_in.npy', training_sequence_in)
np.save('data/data4LSTM/training_sequence_out.npy', training_sequence_out)

np.save('data/data4LSTM/validation_sequence_in.npy', validation_sequence_in)
np.save('data/data4LSTM/validation_sequence_out.npy', validation_sequence_out)

np.save('data/data4LSTM/testing_sequence_in.npy', testing_sequence_in)
np.save('data/data4LSTM/testing_sequence_out.npy', testing_sequence_out)
