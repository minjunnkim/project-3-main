import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    mean = 0.0
    std = 0.0
    total_pixels = 0
    pixel_squared_sum = 0.0

    image_paths = glob.glob(os.path.join(dir_name, "**", "*.*"), recursive=True)
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]

    for path in image_paths:
        img = Image.open(path).convert("L")  
        img_np = np.array(img, dtype=np.float32) / 255.0 

        mean += img_np.sum()
        pixel_squared_sum += (img_np ** 2).sum()
        total_pixels += img_np.size

    mean = mean / total_pixels
    variance = (pixel_squared_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
