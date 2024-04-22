import numpy as np
import matplotlib.pyplot as plt
from sthom215_problem_1_ex_1 import TIFF_IO


class SummaryImages(TIFF_IO):
    def __init__(self, path: str, fnorm: bool = False):
        super().__init__(path)
        self.read_tiff()
        if fnorm:
            self.normalization()
        self.mean = None
        self.median = None
        self.variance = None

    def normalization(self):
        """
        Normalize fluouresnce in the movie by dividing by the minimum value.
        """
        norm_factor = (self.movie.max() - self.movie.min())
        self.movie = (self.movie - self.movie.min()) / norm_factor

    def get_mean(self):
        """
        Get the mean image of the movie.
        Assuming the largest axis is the time axis.
        """
        axis = np.argmax(self.movie.shape)
        return np.mean(self.movie, axis)

    def get_median(self):
        """
        Get the median image of the movie.
        Assuming the largest axis is the time axis.
        """
        axis = np.argmax(self.movie.shape)
        return np.median(self.movie, axis)

    def get_variance(self):
        """
        Get the variance image of the movie.
        Assuming the largest axis is the time axis.
        """
        axis = np.argmax(self.movie.shape)
        return np.var(self.movie, axis)

    def plot_mean_median_variance(self):
        """
        Plot the mean, median, and variance images of the movie.
        Assuming the largest axis is the time axis.
        """

        fns = [self.get_mean, self.get_median, self.get_variance]
        imgs = [fn() for fn in fns]

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), layout='tight')
        for ii, img in enumerate(imgs):
            title = f"{fns[ii].__name__.split('_')[-1]}"
            title = title.capitalize()
            ax[ii].imshow(img, cmap='gray')
            ax[ii].set_title(f"{title} Image")
            ax[ii].set_axis_off()
        fig.suptitle('Mean, Median, and Variance Images')
        plt.show()
