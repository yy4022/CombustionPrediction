import json
import pickle

import numpy as np

from methods_preprocess import load_PIVdata, load_PLIFdata, crop_data, get_min_max, min_max_scaler, MyDataset

"""
Section 1: this part is used for generating the unshuffled numpy data from original datasets.
NOTE: you can comment this part after using.
"""

# # PART 1: load and process the PIV datasets. (normalize and crop)
# # NOTE: in this case, we need to crop, normalize and split it into two (align with PLIF dataset).
# # 1. load the PIV dataset.
# dataset_PIV, PIV_x_points, PIV_y_points = load_PIVdata('data/IA_PIV.mat')
# dataset_PIV_x = dataset_PIV[0, :, :, :]
# dataset_PIV_y = dataset_PIV[1, :, :, :]
# dataset_PIV_z = dataset_PIV[2, :, :, :]
#
# # 2. crop the dataset
# cropped_PIV_x = crop_data(dataset_PIV_x, PIV_x_points, PIV_y_points)
# cropped_PIV_y = crop_data(dataset_PIV_y, PIV_x_points, PIV_y_points)
# cropped_PIV_z = crop_data(dataset_PIV_z, PIV_x_points, PIV_y_points)
#
# # 3. normalize the dataset
# min_PIV_x, max_PIV_x = get_min_max(cropped_PIV_x)
# min_PIV_y, max_PIV_y = get_min_max(cropped_PIV_y)
# min_PIV_z, max_PIV_z = get_min_max(cropped_PIV_z)
#
# # save the global min and max value
# existing_data = {}
#
# # add new information to the file
# new_data = {
#     'min_PIV_x': float(min_PIV_x),
#     'max_PIV_x': float(max_PIV_x),
#     'min_PIV_y': float(min_PIV_y),
#     'max_PIV_y': float(max_PIV_y),
#     'min_PIV_z': float(min_PIV_z),
#     'max_PIV_z': float(max_PIV_z),
# }
# existing_data.update(new_data)
#
# # save the updated data information
# with open('data/dataset_information_PIV.json', 'w') as file:
#     json.dump(existing_data, file)
#
# # use the min-max scaler to normalize the dataset
# normalized_PIV_x = min_max_scaler(cropped_PIV_x, min_PIV_x, max_PIV_x)
# normalized_PIV_y = min_max_scaler(cropped_PIV_y, min_PIV_y, max_PIV_y)
# normalized_PIV_z = min_max_scaler(cropped_PIV_z, min_PIV_z, max_PIV_z)
#
# # 4. split the dataset into [1, 2500] and [2501, 5000] to save
# np.save('data/unshuffled_numpy_data/IA_PIV_x_1.npy', normalized_PIV_x[0:2500, :, :])
# np.save('data/unshuffled_numpy_data/IA_PIV_y_1.npy', normalized_PIV_y[0:2500, :, :])
# np.save('data/unshuffled_numpy_data/IA_PIV_z_1.npy', normalized_PIV_z[0:2500, :, :])
#
# np.save('data/unshuffled_numpy_data/IA_PIV_x_2.npy', normalized_PIV_x[2500:5001, :, :])
# np.save('data/unshuffled_numpy_data/IA_PIV_y_2.npy', normalized_PIV_y[2500:5001, :, :])
# np.save('data/unshuffled_numpy_data/IA_PIV_z_2.npy', normalized_PIV_z[2500:5001, :, :])
#
# # PART 2. load and process the PLIF dataset. (normalize and crop)
# # 1. provide the basic information of the PLIF files
# files_PLIF = ['data/IA_PLIF_1to2500.mat',
#               'data/IA_PLIF_2501to5000.mat']
# file_num = len(files_PLIF)
#
# # 2. use for loop to get the global extreme value
# for i in range(file_num):
#     # load the PLIF dataset
#     dataset_PLIF, PLIF_x_points, PLIF_y_points = load_PLIFdata(files_PLIF[i])
#
#     # crop the dataset
#     cropped_PLIF = crop_data(dataset_PLIF, PLIF_x_points, PLIF_y_points)
#
#     # get the global max and min value
#     min_PLIF, max_PLIF = get_min_max(cropped_PLIF)
#
#     # compare the min and max value with the saved one
#     # try to load the existing file
#     try:
#         # a) if the values have existed, compare and update the value
#         with open('data/dataset_information_PLIF.json', 'r') as file:
#             existing_data = json.load(file)
#
#         current_min_PLIF = existing_data['min_PLIF']
#         current_max_PLIF = existing_data['max_PLIF']
#
#         if current_min_PLIF > float(min_PLIF):
#             existing_data['min_PLIF'] = float(min_PLIF)
#         if current_max_PLIF < float(min_PLIF):
#             existing_data['max_PLIF'] = float(min_PLIF)
#
#     except FileNotFoundError:
#         # b) if the values have not existed, create a new one
#         existing_data = {}
#
#         # add new information to the file
#         new_data = {
#             'min_PLIF': float(min_PLIF),
#             'max_PLIF': float(max_PLIF),
#         }
#
#         existing_data.update(new_data)
#
#     # save the updated data information
#     with open('data/dataset_information_PLIF.json', 'w') as file:
#         json.dump(existing_data, file)
#
#
# # 3. process the PLIF datasets
# for i in range(file_num):
#     # load the PLIF dataset
#     dataset_PLIF, PLIF_x_points, PLIF_y_points = load_PLIFdata(files_PLIF[i])
#
#     # crop the dataset
#     cropped_PLIF = crop_data(dataset_PLIF, PLIF_x_points, PLIF_y_points)
#
#     # normalize the dataset
#     # 1. get the global min and max value
#     with open('data/dataset_information_PLIF.json', 'r') as file:
#         existing_data = json.load(file)
#
#     min_PLIF = np.float32(existing_data['min_PLIF'])
#     max_PLIF = np.float32(existing_data['max_PLIF'])
#
#     # normalize and discretize the datasets according to the min, max values
#     normalized_PLIF = min_max_scaler(cropped_PLIF, min_PLIF, max_PLIF)
#
#     # save this specified data
#     np.save(f'data/unshuffled_numpy_data/IA_PLIF_{i+1}.npy', normalized_PLIF)

"""
Section 2: split the dataset into training, validation and testing datasets (8:1:1).
NOTE: 
1. the training dataset should be shuffled, others should not.
2. the PLIF and PIV datasets should be shuffled with the same shuffled index.
"""

# 1. provide the essential information.
file_num = 2

for i in range(file_num):
    # 2. load the datasets
    PLIF_data = np.load(f'data/unshuffled_numpy_data/IA_PLIF_{i+1}.npy')
    PIV_x_data = np.load(f'data/unshuffled_numpy_data/IA_PIV_x_{i+1}.npy')
    PIV_y_data = np.load(f'data/unshuffled_numpy_data/IA_PIV_y_{i + 1}.npy')
    PIV_z_data = np.load(f'data/unshuffled_numpy_data/IA_PIV_z_{i + 1}.npy')

    # 3. split the dataset with the ratio of (8:1:1)
    data_num = PLIF_data.shape[0]
    split_points = [int(np.floor(data_num * 0.8)), int(np.floor(data_num * 0.9))]

    PLIF_data_split = np.split(PLIF_data, split_points, axis=0)
    PIV_x_data_split = np.split(PIV_x_data, split_points, axis=0)
    PIV_y_data_split = np.split(PIV_y_data, split_points, axis=0)
    PIV_z_data_split = np.split(PIV_z_data, split_points, axis=0)

    training_PLIF_data = PLIF_data_split[0]
    validation_PLIF_data = PLIF_data_split[1]
    testing_PLIF_data = PLIF_data_split[2]

    training_PIV_x_data = PIV_x_data_split[0]
    validation_PIV_x_data = PIV_x_data_split[1]
    testing_PIV_x_data = PIV_x_data_split[2]

    training_PIV_y_data = PIV_y_data_split[0]
    validation_PIV_y_data = PIV_y_data_split[1]
    testing_PIV_y_data = PIV_y_data_split[2]

    training_PIV_z_data = PIV_z_data_split[0]
    validation_PIV_z_data = PIV_z_data_split[1]
    testing_PIV_z_data = PIV_z_data_split[2]

    # 4. shuffle the training datasets and save
    training_num = training_PLIF_data.shape[0]
    index_array = np.arange(training_num)
    np.random.shuffle(index_array)

    training_PLIF_data = np.take(training_PLIF_data, index_array, axis=0)
    training_PIV_x_data = np.take(training_PIV_x_data, index_array, axis=0)
    training_PIV_y_data = np.take(training_PIV_y_data, index_array, axis=0)
    training_PIV_z_data = np.take(training_PIV_z_data, index_array, axis=0)

    training_PLIF_data = np.expand_dims(training_PLIF_data, axis=1)
    training_PIV_x_data = np.expand_dims(training_PIV_x_data, axis=1)
    training_PIV_y_data = np.expand_dims(training_PIV_y_data, axis=1)
    training_PIV_z_data = np.expand_dims(training_PIV_z_data, axis=1)

    training_PLIF_dataset = MyDataset(training_PLIF_data)
    training_PIV_x_dataset = MyDataset(training_PIV_x_data)
    training_PIV_y_dataset = MyDataset(training_PIV_y_data)
    training_PIV_z_dataset = MyDataset(training_PIV_z_data)

    with open(f'data/data4models/training_PLIF_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(training_PLIF_dataset, file)

    with open(f'data/data4models/training_PIV_x_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(training_PIV_x_dataset, file)

    with open(f'data/data4models/training_PIV_y_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(training_PIV_y_dataset, file)

    with open(f'data/data4models/training_PIV_z_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(training_PIV_z_dataset, file)

    # 5. reshape and save the validation datasets
    validation_PLIF_data = np.expand_dims(validation_PLIF_data, axis=1)
    validation_PIV_x_data = np.expand_dims(validation_PIV_x_data, axis=1)
    validation_PIV_y_data = np.expand_dims(validation_PIV_y_data, axis=1)
    validation_PIV_z_data = np.expand_dims(validation_PIV_z_data, axis=1)

    validation_PLIF_dataset = MyDataset(validation_PLIF_data)
    validation_PIV_x_dataset = MyDataset(validation_PIV_x_data)
    validation_PIV_y_dataset = MyDataset(validation_PIV_y_data)
    validation_PIV_z_dataset = MyDataset(validation_PIV_z_data)

    with open(f'data/data4models/validation_PLIF_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(validation_PLIF_dataset, file)

    with open(f'data/data4models/validation_PIV_x_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(validation_PIV_x_dataset, file)

    with open(f'data/data4models/validation_PIV_y_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(validation_PIV_y_dataset, file)

    with open(f'data/data4models/validation_PIV_z_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(validation_PIV_z_dataset, file)

    # 6. save the testing numpy array, reshape and save the testing datasets
    np.save(f'data/data4models/validation_PLIF_data{i + 1}.npy', testing_PLIF_data)
    np.save(f'data/data4models/validation_PIV_x_data{i + 1}.npy', testing_PIV_x_data)
    np.save(f'data/data4models/validation_PIV_y_data{i + 1}.npy', testing_PIV_y_data)
    np.save(f'data/data4models/validation_PIV_z_data{i + 1}.npy', testing_PIV_z_data)

    testing_PLIF_data = np.expand_dims(testing_PLIF_data, axis=1)
    testing_PIV_x_data = np.expand_dims(training_PIV_x_data, axis=1)
    testing_PIV_y_data = np.expand_dims(training_PIV_y_data, axis=1)
    testing_PIV_z_data = np.expand_dims(training_PIV_z_data, axis=1)

    testing_PLIF_dataset = MyDataset(testing_PLIF_data)
    testing_PIV_x_dataset = MyDataset(testing_PIV_x_data)
    testing_PIV_y_dataset = MyDataset(testing_PIV_y_data)
    testing_PIV_z_dataset = MyDataset(testing_PIV_z_data)

    with open(f'data/data4models/testing_PLIF_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(testing_PLIF_dataset, file)

    with open(f'data/data4models/testing_PIV_x_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(testing_PIV_x_dataset, file)

    with open(f'data/data4models/testing_PIV_y_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(testing_PIV_y_dataset, file)

    with open(f'data/data4models/testing_PIV_z_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(testing_PIV_z_dataset, file)



