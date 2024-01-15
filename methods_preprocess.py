from typing import Tuple, List

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


# Define the class of Dataset
class MyDataset(Dataset):
    def __init__(self, img_data):
        self.img_data = img_data
        self.length = len(self.img_data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.img_data[index]

def load_PIVdata(file_PIV: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Load the PIV dataset:
    :param file_PIV: A string represents the filename of PIV image
    :return: A numpy array represents the value of PIV image,
            a numpy array represents the value of x range of PIV image,
            a numpy array represents the value of y range of PIV image.
    """

    with h5py.File(file_PIV, 'r', rdcc_nbytes=1024**3, rdcc_nslots=1) as file:
        """
        Get the PIV numpy array, note that:
            1 denotes the axial(x) velocity,
            2 denotes the radial(y) velocity,
            3 denotes the tangential(z) velocity
        """
        dataset_PIV = file['PIV']['velfield'][:]

        # get the x range of PIV image
        PIV_x = file['PIV']['x'][:]

        # get the y range of PIV image with (73, 1)
        PIV_y = file['PIV']['y'][:]

    return dataset_PIV, PIV_x, PIV_y


def load_PLIFdata(file_PLIF: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Load the PLIF dataset:
    :param file_PLIF: A string represents the filename of PLIF image
    :return: A numpy array represents the value of PLIF image,
            a numpy array represents the value of x range of PLIF image,
            a numpy array represents the value of y range of PLIF image.
    """

    with h5py.File(file_PLIF, 'r', rdcc_nbytes=1024**3, rdcc_nslots=1) as file:
        # get the PLIF numpy array
        dataset_PLIF = file['PLIF']['PLIFfield'][:]

        # get the x range of PLIF image
        PLIF_x = file['PLIF']['x'][:]

        # get the y range of PLIF image
        PLIF_y = file['PLIF']['y'][:]

    return dataset_PLIF, PLIF_x, PLIF_y

def crop_data(image_data: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) \
        -> np.ndarray:

    """
    Internal function:
        Crop the input image data based on specified x and y axis range.

    :param image_data: A numpy array contains the input image data.
    :param x_axis: A numpy array contains the x-axis values corresponding the columns of the image.
    :param y_axis: A numpy array contains the y-axis values corresponding the rows of the image.
    :return: A numpy array contains the cropped image data.
    """

    # STEP 1. define the range of x, y
    cropped_xmin = -18
    cropped_xmax = 18
    cropped_ymin = 0
    cropped_ymax = 36

    # STEP 2. get the indices satisfied the range
    indices_x = np.where((x_axis >= cropped_xmin) & (x_axis <= cropped_xmax))[0]
    indices_y = np.where((y_axis >= cropped_ymin) & (y_axis <= cropped_ymax))[0]

    # STEP 3. crop the dataset via the range
    cropped_data = image_data[:, indices_y[:, np.newaxis], indices_x]

    # STEP 4: change the type of dataset from 'float64' to 'float32'
    cropped_data = cropped_data.astype('float32')

    return cropped_data

def get_min_max(data_list: List[np.ndarray]) -> Tuple[float, float]:

    """
    Get the minimum and maximum values across a list of numpy arrays.

    :param data_list: A list of numpy arrays for which to find the minimum and maximum values.
    :return: Tuple contains two float number representing the minimum and maximum values.
    """

    min_value = 1000
    max_value = -1000

    for data in data_list:
        if np.amin(data) < min_value:
            min_value = np.amin(data)

        if np.amax(data) > max_value:
            max_value = np.amax(data)

    return min_value, max_value

def min_max_scaler(data: np.ndarray, min_value: float, max_value: float) -> np.ndarray:

    """
    Internal function:
        Use the Min-Max scaling to the given data.

    :param data: A numpy array contains the data to be scaled.
    :param min_value: A float contains the minimum value of the data.
    :param max_value: A float contains the maximum value of the data.
    :return: A numpy array of the same shape as 'data', but scaled such that
            its elements lie in the range [0, 1].
    """

    normalized_data = (data - min_value) / (max_value - min_value)
    normalized_data = normalized_data.astype('float32')
    return normalized_data

def split_dataset_overlap(my_dataset: torch.Tensor, n_step_in: int, n_step_out: int):

    """
    Split the given dataset into overlapping sequences of inputs and corresponding sequences of outputs.

    :param my_dataset: a Tensor representing the dataset to be split into sequences.
    :param n_step_in: an integer representing the number of time steps in each sequence.
    :param n_step_out: an integer representing the number of time steps in each output sequence.
    :return: Tuple[List[ndarray], List[ndarray]]: Returns two lists:
        - The first list contains the sequences of inputs.
        - The second list contains the corresponding sequences of outputs.
    """

    my_dataset = my_dataset.cpu().data.numpy()

    sequence_in = []
    sequence_out = []

    for i in range(len(my_dataset)):
        # find the end of sequence_in, sequence_out
        in_end_ix = i + n_step_in
        out_end_ix = in_end_ix + n_step_out

        # check if the end of sequence_out beyond the dataset
        if out_end_ix > len(my_dataset):
            break

        # gather input and output parts of the pattern
        segment_in, segment_out = my_dataset[i:in_end_ix, :], my_dataset[in_end_ix:out_end_ix, :]
        sequence_in.append(segment_in)
        sequence_out.append(segment_out)

    np.array(sequence_in)
    np.array(sequence_out)
    return sequence_in, sequence_out

def split_dataset(my_dataset, n_steps_in, n_steps_out):

    my_dataset = my_dataset.cpu().data.numpy()

    sequence_in = []
    sequence_out = []

    for i in range(len(my_dataset)):
        # find the end of sequence_in, sequence_out
        in_end_ix = i * n_steps_in + n_steps_in
        out_end_ix = in_end_ix + n_steps_out

        # check if the end of sequence_out beyond the dataset
        if out_end_ix > len(my_dataset):
            break

        # gather input and output parts of the pattern
        segment_in, segment_out = my_dataset[i * n_steps_in:in_end_ix, :], my_dataset[in_end_ix:out_end_ix, :]
        sequence_in.append(segment_in)
        sequence_out.append(segment_out)

    np.array(sequence_in)
    np.array(sequence_out)
    return sequence_in, sequence_out


