from PIL import Image

import numpy as np
from numpy.typing import NDArray


################### Constants ###################
KERNELS = {"l_sobel": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
           "r_sobel":  np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
           "l_prewitt": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
           "r_prewitt": np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
           "l_scharr": np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]]) / 16,
           "r_scharr": np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]) / 16
           }


################### ROI Extractor Object ###################

class ROI_Extraction:
    # TODO add seed pixel
    def __init__(self, image: NDArray, kernel: str = None):
        """
        Initialize the ROI_Extraction object.

        Args:
            image (ndarray): The input image.
            kernel (str, optional): The kernel to be used for convolution. Defaults to None.

        Raises:
            ValueError: If the image is None.

        Attributes:
            image (ndarray): The input image.
            mask (ndarray): The mask obtained from the image using the specified kernel and threshold.
            kernel (ndarray): The kernel used for convolution.
            threshold (float): The threshold value calculated from the convolved image.
            __sizeof__ (int): The size of the image.
            __str__ (str): The string representation of the object.
            __repr__ (str): The string representation of the object.

        """
        if image is None:
            raise ValueError("Image is required")
        else:
            self.image = image
        self.mask = self.getmask(image, self.kernel, self.threshold,)
        self.kernel = KERNELS[kernel]
        self.threshold = np.mean(self.convolve_image(self.kernel))
        self.__sizeof__ = image.size
        self.__str__ = "ROI_Extraction"
        self.__repr__ = "ROI_Extraction_Object"

    def get_roi(self, threshold: float, *, method: str = "same") -> NDArray:
        """
        Calculates the region of interest (ROI) based on the given threshold and method.

        Parameters:
            threshold (float): The threshold value used to determine the ROI.
            method (str, optional): The method used to calculate the ROI. Defaults to "same".

        Returns:
            NDArray: The ROI of the image.

        """
        self.get_mask(self.image, self.kernel, threshold, method)
        return self.image * self.mask

    def convolve_image(self, kernel: NDArray, *, mode: str = "same") -> NDArray:
        """
        Convolves the image with the specified kernel.

        Args:
            kernel (NDArray): The kernel to convolve the image with.
            mode (str, optional): The mode of convolution. Defaults to "same".

        Returns:
            NDArray: The convolved image.
        """
        return self.convolve2d(self.image, KERNELS[kernel], mode=mode)

    def get_mask(self, image: NDArray, kernel: NDArray, threshold: float, *, mode: str = "same") -> NDArray:
        """
        Applies a convolution with the given kernel to the input image and returns a binary mask based on a threshold.

        Parameters:
            image (NDArray): The input image.
            kernel (NDArray): The convolution kernel.
            threshold (float): The threshold value for binarizing the convolved image.
            mode (str, optional): The mode for handling borders. Defaults to "same".

        Returns:
            NDArray: The binary mask where values above the threshold are set to 1 and values below or equal to the threshold are set to 0.
        """
        # apply the convolution with kernel
        convolved_img = self.convolve2d(image, kernel, mode=mode)
        # threshold the image
        return np.where(convolved_img > threshold, 1, 0)

    def get_roi_image(self, threshold: float, method: str = "same") -> NDArray:
        """
        Returns the region of interest (ROI) image based on the given threshold and method.

        Parameters:
            threshold (float): The threshold value used to determine the ROI.
            method (str, optional): The method used to calculate the ROI. Defaults to "same".

        Returns:
            NDArray: The ROI image.

        """
        return self.get_roi(threshold, method)

    def convolve2d(self, image: NDArray, kernel: NDArray, mode='same', boundary='constant', fillvalue=0) -> NDArray:
        """
        Perform 2D convolution on an image using a given kernel.

        Args:
            image (NDArray): The input image to convolve.
            kernel (NDArray): The convolution kernel.
            mode (str, optional): The mode of convolution. Defaults to 'same'.
            boundary (str, optional): The boundary condition for padding. Defaults to 'constant'.
            fillvalue (int, optional): The value used for padding. Defaults to 0.

        Returns:
            NDArray: The convolved image.

        Raises:
            ValueError: If the mode is not one of 'valid', 'same', or 'full'.

        """
        # Get dimensions of the image and kernel
        m, n = image.shape
        k, l = kernel.shape

        # Pad the image based on the convolution mode
        if mode == 'same':
            pad_height = (k - 1) // 2
            pad_width = (l - 1) // 2
            image = np.pad(image, ((pad_height, pad_height), (pad_width,
                           pad_width)), mode=boundary, constant_values=fillvalue)
        elif mode == 'full':
            pad_height = k - 1
            pad_width = l - 1
            image = np.pad(image, ((pad_height, pad_height), (pad_width,
                           pad_width)), mode=boundary, constant_values=fillvalue)

        # Flip the kernel
        kernel = np.flipud(np.fliplr(kernel))

        # Perform 1D convolutions along rows and columns
        output = np.zeros_like(image)
        for i in range(m):
            for j in range(n):
                output[i, j] = np.sum(image[i:i+k, j:j+l] * kernel)

        if mode == 'valid':
            return output[k - 1:m - k + 1, l - 1:n - l + 1]
        elif mode == 'same':
            return output[pad_height:pad_height+m, pad_width:pad_width+n]
        elif mode == 'full':
            return output

    def get_available_kernels(self) -> None:
        """
        Prints the available kernels.

        This method iterates over the keys of the KERNELS dictionary and prints each key, representing an available kernel.

        Parameters:
            None

        Returns:
            None
        """
        for key in KERNELS.keys():
            print("the available kernels are:\n", flush=True)
            print("%s".format(key), flush=True)
