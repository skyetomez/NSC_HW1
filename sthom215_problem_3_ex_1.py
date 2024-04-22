import numpy as np
from numpy.typing import NDArray


################### ROI Extractor solution ###################
import numpy as np
from scipy.ndimage import sobel, binary_dilation, label
from scipy import ndimage as ndi


########### MY SOLUTION TO NUM 3 ####################

def get_hw3_1_solution(movie, frame, kernel_dim, seed_pixel):

    selected_frame = movie[frame]
    edge_x = sobel(selected_frame, axis=0, mode='constant')
    edge_y = sobel(selected_frame, axis=1, mode='constant')
    magnitude = np.hypot(edge_x, edge_y)

    # Replace np.mean with a median_of_means function, assuming it's defined
    mom_val = median_of_means(magnitude)
    binary_edges = magnitude > mom_val

    # Initial labeling of all edges
    labeled_array, _ = label(binary_edges)
    seed_label = labeled_array[seed_pixel[0], seed_pixel[1]]

    # Create mask for the region of interest using the seed label
    roi_mask = labeled_array == seed_label

    # Apply dilation only to the selected ROI mask
    structuring_element = np.ones((kernel_dim, kernel_dim))
    dilated_mask = binary_dilation(roi_mask, structure=structuring_element)

    return dilated_mask


def median_of_means(movie, ss: int = 300):
    means = []
    for _ in range(7):
        indexes = np.random.choice(movie.shape[0], ss)
        mean = np.mean(movie[indexes])
        means.append(mean)
    return np.median(means, axis=0)


########### MY SOLUTION ATTEMPT AT DOING FROM SCRATCH ####################

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
    def __init__(self, image: NDArray, kernel: str = None):
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
        self.get_mask(self.image, self.kernel, threshold, method)
        return self.image * self.mask

    def convolve_image(self, kernel: NDArray, *, mode: str = "same") -> NDArray:
        return self.convolve2d(self.image, KERNELS[kernel], mode=mode)

    def get_mask(self, image: NDArray, kernel: NDArray, threshold: float, *, mode: str = "same") -> NDArray:
        convolved_img = self.convolve2d(image, kernel, mode=mode)
        return np.where(convolved_img > threshold, 1, 0)

    def get_roi_image(self, threshold: float, method: str = "same") -> NDArray:
        return self.get_roi(threshold, method)

    def convolve2d(self, image: NDArray, kernel: NDArray, mode='same', boundary='constant', fillvalue=0) -> NDArray:
        m, n = image.shape
        k, l = kernel.shape

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

        kernel = np.flipud(np.fliplr(kernel))

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
        for key in KERNELS.keys():
            print("the available kernels are:\n", flush=True)
            print("%s".format(key), flush=True)


######################## DEPRECEATED  #####################

# def get_hw3_1_solution(movie, frame, kernel_dim, seed_pixel):

#     selected_frame = movie[frame]
#     edge_x = sobel(selected_frame, axis=0, mode='constant')
#     edge_y = sobel(selected_frame, axis=1, mode='constant')
#     magnitude = np.hypot(edge_x, edge_y)

#     # mean_val = np.mean(magnitude)
#     mean_val = median_of_means(magnitude)
#     binary_edges = magnitude > mean_val

#     structuring_element = np.ones((kernel_dim, kernel_dim))
#     dilated_mask = binary_dilation(binary_edges, structure=structuring_element)

#     labeled_array, _ = label(dilated_mask)
#     seed_label = labeled_array[seed_pixel[0], seed_pixel[1]]
#     roi_mask = labeled_array == seed_label

#     return roi_mask
