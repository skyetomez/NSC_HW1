import numpy as np
from numpy.typing import NDArray


###################### CROSS-CORRELATION ######################

class XCORR:
    def __init__(self, frame1: NDArray, frame2: NDArray):
        self.frame1 = frame1
        self.frame2 = frame2
        self.xcorr = 0.0

    def get_xcorr_fft(self):
        fft_frame1 = np.fft.fft2(self.frame1)
        fft_frame2 = np.fft.fft2(self.frame2)
        xcorr_freq = fft_frame1 * np.conj(fft_frame2)
        xcorr = np.fft.ifft2(xcorr_freq)
        energy1 = np.sum(np.abs(self.frame1)**2)
        energy2 = np.sum(np.abs(self.frame2)**2)
        total_energy = np.sqrt(energy1 * energy2)
        self.xcorr = np.abs(np.fft.fftshift(xcorr)) / total_energy

    def get_peak(self):
        return np.unravel_index(np.argmax(self.xcorr), self.xcorr.shape)
