from sthom215_problem_3_ex_1 import ROI_Extraction
import numpy as np
from numpy.typing import NDArray
import numpy as np
from skimage import io


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


def create_mask_for_bounding_box(movie, bounding_box):
    """
    Create a mask for the given bounding box and apply it across all sections of a matrix with shape (t, m, m).

    Parameters:
    - image: The 2D numpy array representing the image.
    - bounding_box: A tuple (x, y, width, height) representing the bounding box.
    - matrix_shape: The shape of the 3D matrix (t, m, m) where the mask will be applied.

    Returns:
    - A 3D numpy array with the mask applied along axis 0.
    """

    # Unpack the bounding box coordinates and dimensions
    x, y, width, height = bounding_box

    # Create a 2D mask with the same dimensions as the image
    mask_2d = np.zeros_like(movie[0, :, :])

    # Set the region within the bounding box to 1 (mask on)
    mask_2d[y:y+height, x:x+width] = 1

    # Initialize a 3D matrix based on the provided shape
    t, m, m = movie.shape
    # mask_3d = np.zeros((t, m, m))

    # Apply the 2D mask across all 't' slices along axis 0

    return np.multiply(movie, mask_2d.reshape(1, m, m)), mask_2d


def extract_time_traces(movie, rois):
    """
    Function to extract time-traces for ROIs identified in Problem 2 from a TIFF movie.

    Parameters:
    - movie: numpy array of shape (num_frames, height, width), representing the TIFF movie.
    - rois: list of numpy arrays, each representing an ROI identified in Problem 2.

    Returns:
    - time_traces: list of numpy arrays, each representing the time-trace for an ROI.
    """
    num_frames, height, width = movie.shape
    num_rois = len(rois)
    time_traces = []

    for roi in rois:
        roi_trace = []
        for frame in movie:
            # Calculate the average intensity within the ROI for each frame
            roi_intensity = np.mean(frame[roi])
            roi_trace.append(roi_intensity)
        time_traces.append(np.array(roi_trace))

    return time_traces


#################### Time-Trace Estimation ################


class TimeTraceEstimation(ROI_Extraction):
    def __init__(self, image: NDArray, mask=None, kernel=None):
        super().__init__(image, mask, kernel)

    def apply_kernel(self, kernel: NDArray) -> NDArray:
        return np.convolve(self.image, kernel, mode='same')

    def estimate_time_trace(self) -> NDArray:
        if self.kernel is None:
            return self.image
        else:
            return self.apply_kernel(self.kernel)
