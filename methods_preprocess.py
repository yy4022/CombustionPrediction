from typing import Tuple

import numpy as np
import h5py


def load_PIVdata(file_PIV: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Load the PIV dataset:
    :param file_PIV: A string represents the filename of PIV image
    :return: A numpy array represents the value of PIV image,
            a numpy array represents the value of x range of PIV image,
            a numpy array represents the value of y range of PIV image.
    """

    with h5py.File(file_PIV, 'r') as file:
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

    with h5py.File(file_PLIF, 'r') as file:
        # get the PLIF numpy array with the shape of (1689, 409, 658)
        dataset_PLIF = file['PLIF']['PLIFfield'][:]

        # get the x range of PLIF image with (658, 1)
        PLIF_x = file['PLIF']['x'][:]

        # get the y range of PLIF image with (409, 1)
        PLIF_y = file['PLIF']['y'][:]

    return dataset_PLIF, PLIF_x, PLIF_y