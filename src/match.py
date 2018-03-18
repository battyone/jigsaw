#!/usr/bin/env python
"""
Functionality for determining whether two puzzle pieces match.

Usage:
    match.py show <filepath1> <filepath2>
"""

import docopt
import numpy as np
from matplotlib import pyplot

import piece


def _smooth(signal, cutoff, d=1):
    """Smooth a signal through a high pass filter.

    :param signal: An np.array representing the signal.
    :param cutoff: The frequency above which all frequencies will be removed.
    :param d:
        The sampling rate used to acquire the signal. Makes the cutoff
        parameter meaningful.
    :return:
        The original signal with the high frequencies removed.
    """
    f = np.fft.rfft(signal)
    fx = np.fft.rfftfreq(n=signal.size, d=d)
    f[fx > cutoff] = 0
    return np.fft.irfft(f)


def _smooth_contour(c, cutoff=3.5e-3):
    c1 = c[:, 0]
    c2 = c[:, 1]
    c1f = _smooth(c1, cutoff)
    c2f = _smooth(c2, cutoff)
    return np.vstack([c1f, c2f]).T


def _show_match(filepath1, filepath2):
    p1 = piece.read_piece(filepath1)
    p2 = piece.read_piece(filepath2)
    pyplot.subplot(121)
    pyplot.plot(p1.contour[:, 0], p1.contour[:, 1])
    pyplot.axis('equal')
    pyplot.subplot(122)
    pyplot.plot(p2.contour[:, 0], p2.contour[:, 1])
    pyplot.axis('equal')
    pyplot.suptitle('Matching: {} with {}'.format(
        filepath1, filepath2))
    pyplot.show()


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    if args['show']:
        _show_match(args['<filepath1>'], args['<filepath2>'])
