#!/usr/bin/env python
"""
A collection of math utilities.
"""

import numpy as np


def get_rotation_matrix_2d(angle):
    return np.array(
        [[np.cos(angle), np.sin(angle)],
         [-np.sin(angle), np.cos(angle)]]
    )


def rotate_2d(points, angle):
    matrix = get_rotation_matrix_2d(angle)
    return np.matmul(matrix, points.T).T
