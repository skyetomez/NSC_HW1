import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from typing import Tuple
import numpy as np


###################### XCORR VECTORIZED ######################

def get_correlation(frame1: NDArray, frame2: NDArray) -> Tuple[float, int]:
    """
    Done vectorized, very slow and sliding;
    """
    correlations = []
    frame1_flat = frame1.flatten()
    frame2_flat = frame2.flatten()
    # Loop from negative shift to positive shift to cover all possible alignments
    max_shift = len(frame1_flat)
    for slide in range(-max_shift + 1, max_shift):
        if slide < 0:
            correlation = np.correlate(
                frame1_flat[:slide], frame2_flat[-slide:])
        else:
            correlation = np.correlate(
                frame1_flat[slide:], frame2_flat[:-slide] if slide != 0 else frame2_flat)
        # Append the scalar result of correlation
        correlations.append(correlation[0])
    # Find the index of the maximum correlation and adjust it to match the correct shift
    max_correlation = max(correlations)
    best_shift = np.argmax(correlations) - max_shift + 1
    return max_correlation, best_shift


###################### CROSS-CORRELATION ######################


class XCORR:
    def __init__(self, frame1: NDArray, frame2: NDArray):
        self.frame1 = frame1
        self.frame2 = frame2
        self.xcorr = 0.0

    def get_xcorr_fft(self) -> Tuple[int, int]:
        fft_frame1 = np.fft.fft2(self.frame1)
        fft_frame2 = np.fft.fft2(self.frame2)
        xcorr_freq = fft_frame1 * np.conj(fft_frame2)
        xcorr = np.fft.ifft2(xcorr_freq)
        energy1 = np.sum(np.abs(self.frame1)**2)
        energy2 = np.sum(np.abs(self.frame2)**2)
        total_energy = np.sqrt(energy1 * energy2)
        self.xcorr = np.abs(np.fft.fftshift(xcorr)) / total_energy
        return self.get_peak()

    def get_peak(self) -> Tuple[int, int]:
        return np.unravel_index(np.argmax(self.xcorr), self.xcorr.shape)

    def plot_xcorr_heatmap(self):
        data = go.Heatmap(z=self.xcorr, colorscale='Viridis')
        layout = go.Layout(title='Cross-correlation Heatmap',
                           xaxis_title='X Shift', yaxis_title='Y Shift')
        fig = go.Figure(data=data, layout=layout)
        fig.show()

    def plot_xcorr_surface_3d(self):
        x = np.arange(self.xcorr.shape[1])
        y = np.arange(self.xcorr.shape[0])
        X, Y = np.meshgrid(x, y)
        data = go.Surface(z=self.xcorr, x=X, y=Y, colorscale='Viridis')
        layout = go.Layout(title='3D Cross-correlation Surface', scene={
                           'xaxis_title': 'X Shift', 'yaxis_title': 'Y Shift', 'zaxis_title': 'Correlation'})
        fig = go.Figure(data=data, layout=layout)
        fig.show()
