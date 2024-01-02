from methods_preprocess import load_PIVdata, load_PLIFdata
from methods_show import show_image

"""
This file is used for visualising the PIV and PLIF data, for defining the x, y range of cropped image.
"""

# PART 1: visualise the PIV image.
# 1. load the PIV dataset.
dataset_PIV, PIV_x_points, PIV_y_points = load_PIVdata('data/IA_PIV.mat')

# 2. show the first image of PIV-x, y, z.
PIV_xmin = PIV_x_points.min()
PIV_xmax = PIV_x_points.max()
PIV_ymin = PIV_y_points.min()
PIV_ymax = PIV_y_points.max()

show_image(dataset_PIV[0, 0, :, :], PIV_xmin, PIV_xmax, PIV_ymin, PIV_ymax, 'PIV-x')
show_image(dataset_PIV[1, 0, :, :], PIV_xmin, PIV_xmax, PIV_ymin, PIV_ymax, 'PIV-y')
show_image(dataset_PIV[2, 0, :, :], PIV_xmin, PIV_xmax, PIV_ymin, PIV_ymax, 'PIV-z')

print(PIV_xmin)
print(PIV_xmax)
print(PIV_ymin)
print(PIV_ymax)

# PART 2: visualise the PLIF image.
# 1. load the first PLIF dataset.
dataset_PLIF, PLIF_x_points, PLIF_y_points = load_PLIFdata('data/IA_PLIF_1to2500.mat')

# 2. show the first image of PLIF.
PLIF_xmin = PLIF_x_points.min()
PLIF_xmax = PLIF_x_points.max()
PLIF_ymin = PLIF_y_points.min()
PLIF_ymax = PLIF_y_points.max()

show_image(dataset_PLIF[0, :, :], PLIF_xmin, PLIF_xmax, PLIF_ymin, PLIF_ymax, 'PLIF')

print(PLIF_xmin)
print(PLIF_xmax)
print(PLIF_ymin)
print(PLIF_ymax)

"""
Conclusion: 
according to the result, we can define x-range [-18, 18] and y-range [0, 36].
"""
