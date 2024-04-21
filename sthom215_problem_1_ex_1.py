import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy.typing import NDArray
from celluloid import Camera


#################### I/ O #############################

class TIFF_IO:
    def __init__(self, path: str):
        self.path = path
        self.num_frames = None
        self.movie = None

    def read_tiff(self) -> None:
        self.open_tiff()
        self.num_frames = self.movie.shape[0]
        return self.movie

    def open_tiff(self) -> None:
        # Open the TIFF file in read mode
        max_workers = os.cpu_count() // 2
        self.movie = tifffile.imread(self.path, maxworkers=max_workers)


def save_gif(movie: NDArray, num_frames: int) -> None:
    fig = plt.figure(figsize=(4, 4), tight_layout=True)
    ax = fig.gca()

    camera = Camera(fig)
    for i in np.arange(num_frames):
        ax.imshow(movie[i], cmap='turbo')
        ax.axis('off')
        camera.snap()
    animation = camera.animate()
    animation.save('movie.mp4', fps=60)
