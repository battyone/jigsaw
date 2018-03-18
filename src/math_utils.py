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


def curvature(contour):
    x = contour[:, 0]
    y = contour[:, 1]
    xp = np.gradient(x)
    yp = np.gradient(y)
    xpp = np.gradient(xp)
    ypp = np.gradient(yp)
    return (xp * ypp - yp * xpp) / np.power(xp**2 + yp**2, 3 / 2)


def zero_crossings(x):
    return np.where(np.diff(np.sign(x)))[0]
