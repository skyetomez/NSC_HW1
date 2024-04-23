import numpy as np
from numpy.typing import NDArray
import plotly.express as px
import matplotlib.pyplot as plt


################### ROI Extractor solution ###################
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, prewitt, label
from scipy import ndimage as ndi


########### MY SOLUTION TO NUM 3 ####################

ROI = {
    0: (347, 50, 13, 13),
    1: (113, 140, 11, 11),
    2: (279, 295, 11, 13),
    3: (290, 309, 24, 23),
    4: (348, 323, 7, 6),
    5: (370, 338, 13, 11),
    6: (337, 349, 22, 44),
    7: (288, 367, 33, 25),
    8: (385, 375, 4, 4),
    9: (405, 377, 19, 27),
    10: (381, 413, 11, 18),
    11: (294, 424, 8, 7),
    12: (400, 434, 12, 10),
    13: (257, 453, 7, 8),
    14: (123, 464, 16, 11),
    15: (166, 477, 13, 11)
}


def get_hw3_1_solution(movie):

    print("The ROI masks are as follows: \n", flush=True)
    for frame, seed in ROI.items():
        print(f"Frame: {frame}, Seed: {seed}", flush=True)

    kernel_dim = 3
    for frame, seed in ROI.items():
        roi_mask = get_roi(movie, frame, kernel_dim, seed)
        fig = px.imshow(roi_mask, color_continuous_scale='gray')
        fig.update_layout(coloraxis_showscale=False)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                          title_text=f'roi_mask_{frame}')
        fig.show()


def get_hw3_roi(movie):
    kernel_size = 3
    for frame, seed in ROI.items():
        roi_mask = get_roi(movie, frame, kernel_size, seed)
        np.save(f'roi_mask_{frame}.npy', roi_mask)


def get_roi_2(frame, sigma, kernel_dim):
    tmp2 = gaussian_filter(frame, 3)
    tmp2 = median_filter(tmp2, 7)
    tmp2 = np.where(tmp2 > np.quantile(tmp2.flatten(), .99), 1, 0)
    return tmp2


def get_bounding_boxes(ROIs, plot: bool = False):
    # Check if the image is in grayscale, if not, convert it
    if len(ROIs.shape) > 2:
        # Assuming the image is RGBA or RGB, we convert it to grayscale
        ROIs = np.dot(ROIs[..., :3], [0.2989, 0.5870, 0.1140])

    # Label the regions in the image
    labeled_image, num_features = ndi.label(ROIs)

    # Initialize list for storing bounding boxes
    rois_bounding_boxes = []

    # Find objects and extract bounding boxes
    slices = ndi.find_objects(labeled_image)
    for dy, dx in slices:
        # Get the bounds of the bounding box
        x_start, x_stop = dx.start, dx.stop
        y_start, y_stop = dy.start, dy.stop

        # Adjust coordinates to ensure they stay within the image boundaries
        x_start = max(x_start, 0)
        y_start = max(y_start, 0)
        x_stop = min(x_stop, ROIs.shape[1])
        y_stop = min(y_stop, ROIs.shape[0])

        # Calculate width and height
        width = x_stop - x_start
        height = y_stop - y_start

        # Append adjusted bounding box
        rois_bounding_boxes.append((x_start, y_start, width, height))

    if plot:
        # Visualize the bounding boxes on the original image
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ROIs, cmap='gray', interpolation='nearest')
        for bbox in rois_bounding_boxes:
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, linewidth=1,
                                 edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.show()

    return rois_bounding_boxes


def get_roi(movie, frame, kernel_dim, seed_pixel, sigma=3):

    selected_frame = movie[frame]

    edge = gaussian_filter(selected_frame, sigma=sigma)
    edge = median_filter(edge, size=7)
    thres = np.quantile(edge.flatten(), .99)
    # mom_val = median_of_means(magnitude)

    binary_edges = np.where(edge > thres, 1, 0)
    binary_edges = prewitt(binary_edges)

    # Initial labeling of all edges
    labeled_array, _ = label(binary_edges)
    seed_label = labeled_array[seed_pixel[0], seed_pixel[1]]

    # Create mask for the region of interest using the seed label
    roi_mask = labeled_array == seed_label

    # Apply dilation only to the selected ROI mask
    # structuring_element = np.ones((kernel_dim, kernel_dim))
    # dilated_mask = binary_dilation(roi_mask, structure=structuring_element)

    return roi_mask


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
