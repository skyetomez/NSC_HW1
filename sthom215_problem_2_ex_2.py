import numpy as np
from typing import Tuple
from numpy.typing import NDArray
import matplotlib.pyplot as plt


############## IMAGE SUMMARY STATISTICS ##############


def get_pixelDistribution(movie: NDArray, x_coord: int, y_coord: int, sigma: float) -> None:
    """
    Returns the pixel distribution of a single pixel across the movie.
    """

    def smooth_kernel(x: NDArray, sigma: float) -> NDArray:
        return np.exp(-x**2 / (2 * sigma**2))

    def smooth_data(y: NDArray, sigma: float, resolution: int = 100) -> Tuple[NDArray, NDArray]:
        x_range = np.linspace(int(min(x)), int(
            max(x)), resolution * int((max(x) - min(x))))
        y_smoothed = np.zeros_like(x_range)
        for xi, yi in zip(x, y):
            y_smoothed += yi * smooth_kernel(x_range - xi, sigma)
        return x_range, y_smoothed

    pixels = movie[:, y_coord, x_coord]
    H = np.histogram(pixels, bins=256)
    x = H[1][:-1]
    y = H[0]/np.sum(H[0])
    plt.bar(x, y, linewidth=2, alpha=0.5)
    x_smooth, y_smooth = smooth_data(y, sigma)
    plt.plot(x_smooth, max(y)*y_smooth/max(y_smooth),
             color='red', linewidth=2, label='smoothed')
    plt.title(f'Pixel Distribution at ({x_coord}, {y_coord})')


def get_edge_mean(movie: NDArray) -> NDArray:
    """
    Returns the mean of the edge pixels of the movie.
    """
    axis = np.argmax(movie.shape)
    diff = movie[:, :, 1:] - movie[:, :, :-1]
    return np.mean(diff, axis)
