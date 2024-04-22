from sthom215_problem_3_ex_1 import ROI_Extraction
import numpy as np
from numpy.typing import NDArray
import numpy as np
from skimage import io


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
