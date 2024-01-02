import json

import numpy as np

from methods_preprocess import load_PIVdata, load_PLIFdata, crop_data, get_min_max, min_max_scaler

# PART 1: load and process the PIV datasets. (normalize and crop)
# NOTE: in this case, we need to crop, normalize and split it into two (align with PLIF dataset).
# 1. load the PIV dataset.
dataset_PIV, PIV_x_points, PIV_y_points = load_PIVdata('data/IA_PIV.mat')
dataset_PIV_x = dataset_PIV[0, :, :, :]
dataset_PIV_y = dataset_PIV[1, :, :, :]
dataset_PIV_z = dataset_PIV[2, :, :, :]

# 2. crop the dataset
cropped_PIV_x = crop_data(dataset_PIV_x, PIV_x_points, PIV_y_points)
cropped_PIV_y = crop_data(dataset_PIV_y, PIV_x_points, PIV_y_points)
cropped_PIV_z = crop_data(dataset_PIV_z, PIV_x_points, PIV_y_points)

# 3. normalize the dataset
min_PIV_x, max_PIV_x = get_min_max(cropped_PIV_x)
min_PIV_y, max_PIV_y = get_min_max(cropped_PIV_y)
min_PIV_z, max_PIV_z = get_min_max(cropped_PIV_z)

# save the global min and max value
existing_data = {}

# add new information to the file
new_data = {
    'min_PIV_x': float(min_PIV_x),
    'max_PIV_x': float(max_PIV_x),
    'min_PIV_y': float(min_PIV_y),
    'max_PIV_y': float(max_PIV_y),
    'min_PIV_z': float(min_PIV_z),
    'max_PIV_z': float(max_PIV_z),
}
existing_data.update(new_data)

# save the updated data information
with open('data/dataset_information_PIV.json', 'w') as file:
    json.dump(existing_data, file)

# use the min-max scaler to normalize the dataset
normalized_PIV_x = min_max_scaler(cropped_PIV_x, min_PIV_x, max_PIV_x)
normalized_PIV_y = min_max_scaler(cropped_PIV_y, min_PIV_y, max_PIV_y)
normalized_PIV_z = min_max_scaler(cropped_PIV_z, min_PIV_z, max_PIV_z)

# 4. split the dataset into [1, 2500] and [2501, 5000] to save
np.save('data/unshuffled_numpy_data/IA_PIV_x_1to2500.npy', normalized_PIV_x[0:2500, :, :])
np.save('data/unshuffled_numpy_data/IA_PIV_y_1to2500.npy', normalized_PIV_y[0:2500, :, :])
np.save('data/unshuffled_numpy_data/IA_PIV_z_1to2500.npy', normalized_PIV_z[0:2500, :, :])

np.save('data/unshuffled_numpy_data/IA_PIV_x_2501to5000.npy', normalized_PIV_x[2500:5001, :, :])
np.save('data/unshuffled_numpy_data/IA_PIV_y_2501to5000.npy', normalized_PIV_y[2500:5001, :, :])
np.save('data/unshuffled_numpy_data/IA_PIV_z_2501to5000.npy', normalized_PIV_z[2500:5001, :, :])

# PART 2. load and process the PLIF dataset. (normalize and crop)
files_PLIF = ['data/IA_PLIF_1to2500.mat',
              'data/IA_PLIF_2501to5000.mat']

dataset_PLIF, PLIF_x_points, PLIF_y_points = load_PLIFdata('data/IA_PLIF_1to2500.mat')



