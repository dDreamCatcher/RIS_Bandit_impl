from kernel import kernel_function
from math import sqrt
import numpy as np

def compute_diff_norm(weight1, weight2, sample_points, l, kernel_index):
    """
    @param weight1:
    @param l
    """

    diff_norm_square = 0
    num_points = sample_points.shape[0]
    diff_weight = weight1 - weight2
    sample_points = np.ndarray.flatten(sample_points)
    for i in range(num_points):
        x_i = sample_points[i]
        w_i = diff_weight[i]
        for j in range(num_points):
            x_j = sample_points[j]
            w_j = diff_weight[j]
            diff_norm_square = diff_norm_square + w_i*w_j*kernel_function([x_i], [x_j], l, kernel_index)

    return sqrt(diff_norm_square)