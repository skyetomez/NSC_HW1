from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from celluloid import Camera


#################### I/ O #############################

class TIFF_IO:
    def __init__(self, path: str):
        self.path = path
        self.num_frames = None
        self.movie = None

    def read_tiff(self) -> None:

        # Check the number of layers in the TIFF image
        image_tiff = self.open_tiff()
        self.num_frames = image_tiff.n_frames

        if self.num_frames < 2:
            self.num_frames = 1
            self.movie = np.asarray(image_tiff)
        else:
            layers = []

            # Iterate over each layer and append it to the list
            for i in range(self.num_frames):
                image_tiff.seek(i)
                layers.append(np.array(image_tiff))

            # Convert the list of layers to a NumPy array
            self.movie = np.asarray(layers)

    def open_tiff(self) -> None:
        # Open the TIFF file in read mode
        return Image.open(self.path)


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
