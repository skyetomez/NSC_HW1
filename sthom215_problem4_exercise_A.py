from sthom215_problem_3_exercise_1 import ROI_Extraction
import numpy as np
from numpy.typing import NDArray

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
