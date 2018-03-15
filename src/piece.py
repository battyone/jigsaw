#!/usr/bin/env python
"""
Functionality for processing individual puzzle pieces.

This module contains functions and data structures for handling individual
puzzle pieces. Puzzle pieces are assumed to be imaged against a solid
background color, preferrably a solid green background like the ones used
in hollywood movies.

To extract a puzzle piece first image the background alone and obtain its
representative color using get_average_pixel(). Then call get_puzzle_piece()
to obtain the data structures describing the puzzle piece needed downstream
for further processing.

Usage:
    piece.py show <filepath>
    piece.py slideshow <file_pattern>
"""

import collections
import cv2
import docopt
import glob
from matplotlib import pyplot
import numpy as np
from scipy import ndimage

import math_utils

# The default background color.
GREEN_SCREEN_RGB = np.array([29.35956073, 196.16425271, 135.00635897])

PuzzlePiece = collections.namedtuple(
    'PuzzlePiece', ('image', 'contour', 'mask'))


def get_average_pixel(image):
    """Returns the average pixel in an image.

    This function is meant to be used to obtain a pixel best representing the
    background color against which the puzzle pieces are imaged.

    :param image: A 3-channel image comprised of a uniform background color.
    :return:
        The average pixel in the image.
    :rtype: nd.array
    """
    return np.average(image, axis=(0, 1))


def _locate(image, background_color, low_sensitivity=0.35,
            high_sensitivity=1.3, kernel_size=11):
    """Locates the puzzle piece within the image.

    The function performs background subtraction to find all the pixels
    associated with the puzzle piece.

   :param image: The image containing the puzzle piece against the background.
   :param background_color: The RGB color of the background.
   :param low_sensitivity:
        The multiplicative factor of the background color pixel to serve as
        the upperbound for color.
   :param high_sensitivity:
        The multiplicative factor of the background color pixel to serve as
        the lowerbound of the threshold operation.
    :param int kernel_size:
        The size of the kernel to use for closing holes remaining after the
        background subtraction takes place.
   :return:
        A mask with the same rows and columns as image, containing 1's for
        pixel locations where the puzzle piece exists, and 0 everywhere else.
    """
    low = background_color * low_sensitivity
    high = background_color * high_sensitivity
    mask = cv2.inRange(image, low, high)
    _, mask = cv2.threshold(mask, 100, 1, cv2.THRESH_BINARY_INV)
    labeled_array, num_labels = ndimage.measurements.label(mask)
    histogram, _ = np.histogram(labeled_array, num_labels)
    histogram = histogram[1:]
    piece_label = np.argmax(histogram) + 1
    mask_label = (labeled_array == piece_label).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(mask_label, cv2.MORPH_CLOSE, kernel)


def _extract_contour(mask):
    """Extracts the contour surrounding a mask.

    The function assumes the image has only one contour.

    :param mask:
        A mask containing 1's in the region of interest and 0's everywhere
        else.

    :return: An opencv contour surrounding the mask.
    """
    _, all_contours, _ = cv2.findContours(
        image=mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE)
    contour = all_contours[0].reshape(-1, 2)
    return contour


def _normalize(image, mask, contour):
    """Rotate, center and crop the data structures comprising the puzzle piece.

    :param image: The image containing the puzzle piece.
    :param mask: A mask containing 1's where the pixels belong to the piece.
    :param contour: A contour describing the shape of the puzzle piece.
    :return: A rotated, centered and cropped versions of the arguments.
    :rtype: PuzzlePiece
    """
    corner1, corner2, angle_degrees = cv2.minAreaRect(contour)
    center = np.average([corner1, corner2], axis=0)
    angle = np.deg2rad(angle_degrees)
    contour_rotated = math_utils.rotate_2d(contour - center, angle) + center

    # Rotate
    rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle_degrees, 1)
    num_rows, num_cols, _ = image.shape
    image_rotated = cv2.warpAffine(image, rotation_matrix,
                                   (num_rows * 2, num_cols * 2))

    # Trim
    x, y, width, height = cv2.boundingRect(contour_rotated.astype(np.int32))
    image_trim = image_rotated[y:(y + height), x:(x + width), :]
    contour_trim = (contour_rotated - np.array([x, y])).astype(np.int32)
    mask_trim = np.zeros(np.array([height, width]), dtype=np.int32)
    cv2.drawContours(mask_trim, [contour_trim], 0, 1, cv2.FILLED)

    return PuzzlePiece(image=image_trim, mask=mask_trim,
                       contour=contour_trim)


def _pad_image(image, factor=2):
    """Enlarges the image canvas, filling new pixels with colors at the edge.

    :param image: An RGB image
    :param factor:
        A scalar representing the factor in which the canvas is to be enlarged.
        A factor of 2 will double the canvas, a factor of 1 will keep it
        unchanged.
    :return:
        A padded version of the image provided as input, with the canvas size
        enlarged by the provided factor, and the original image placed in its
        center.
    """
    if factor < 1:
        raise ValueError('must pad image by a factor >= 1.')
    num_rows = image.shape[0]
    num_cols = image.shape[1]
    pad_rows = int(num_rows * (factor - 1)) // 2
    pad_cols = int(num_cols * (factor - 1)) // 2
    padding = [(pad_rows, pad_rows), (pad_cols, pad_cols)]
    if len(image.shape) == 3:
        padding.append((0, 0))
    return np.pad(image, padding, mode='edge')


def get_puzzle_piece(image, background=GREEN_SCREEN_RGB):
    """Process a raw image of a puzzle piece against a background.

    The processes
    :param image:
        The RGB image containing the puzzle piece against the background.
    :param background_color: The RGB color of the background.
    :return:
        A PuzzlePiece containing useful data structures describing the puzzle
        piece.
    :rtype: PuzzlePiece
    """
    padded_image = _pad_image(image, factor=1.2)
    mask = _locate(padded_image, background)
    contour = _extract_contour(mask)
    return _normalize(padded_image, mask, contour)


def visualize_piece(piece):
    """Visualize the data structures contained in a PuzzlePiece.

    :param piece: a PuzzlePiece named tuple.
    """
    pyplot.subplot(221)
    pyplot.title('Image')
    pyplot.imshow(piece.image)
    pyplot.subplot(222)
    pyplot.title('Contour')
    cv2.drawContours(piece.image, [piece.contour], 0, (0, 0, 255), 10)
    pyplot.imshow(piece.image)
    pyplot.subplot(223)
    pyplot.title('Mask')
    pyplot.imshow(piece.mask)


def read_piece(filepath):
    """Extract a puzzle piece from a file in the filesystem.

    :param str filepath:
        A path to a file in the filesystem containing an image of a puzzle
        piece against a uniform background.
    :return: A PuzzlePiece namedtuple extracted from the file.
    :rtype: PuzzlePiece
    """
    image = cv2.imread(filepath)
    return get_puzzle_piece(image)


def _show_piece(filepath):
    """Plot a puzzle piece from a file."""
    piece = read_piece(filepath)
    pyplot.figure(figsize=(10, 10))
    pyplot.suptitle(filepath)
    visualize_piece(piece)
    pyplot.show()


def _slideshow(file_pattern):
    """Shows a sequence of puzzle_pieces for files matching a glob pattern.

    This function is useful for quickly skimming through a large number of
    puzzle piece images to inspect the reuslts of the methods in this module.
    It accepts a glob, i.e. "data/unicorn/??.jpg" and processes each matching
    file, showing the resulting puzzle piece data structures. When the user
    clicks a mouse button, the function procceeds to plotting the next file.
    """
    for filepath in sorted(glob.glob(file_pattern)):
        piece = read_piece(filepath)
        pyplot.figure(1, figsize=(10, 10))
        pyplot.clf()
        pyplot.suptitle(filepath)
        visualize_piece(piece)
        pyplot.draw()
        pyplot.waitforbuttonpress()
    pyplot.close()


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    if args['show']:
        _show_piece(args['<filepath>'])
    elif args['slideshow']:
        _slideshow(args['<file_pattern>'])
