import numpy as np
from scipy.fft import fft, ifft

def high_pass_filter(signal, cutoff_ratio=0.5):
    transformed_signal = fft(signal)
    mask = np.ones(signal.shape)
    mask[:int(cutoff_ratio * len(signal))] = 0
    filtered_signal = transformed_signal * mask
    denoised_signal = np.real(ifft(filtered_signal))
    return denoised_signal

def low_pass_filter(signal, cutoff_ratio=0.5):
    transformed_signal = fft(signal)
    mask = np.ones(signal.shape)
    mask[int(cutoff_ratio * len(signal)):] = 0
    filtered_signal = transformed_signal * mask
    denoised_signal = np.real(ifft(filtered_signal))
    return denoised_signal

def band_stop_filter(signal, lower_cutoff_ratio=0.25, higher_cutoff_ratio=0.75):
    transformed_signal = fft(signal)
    mask = np.ones(signal.shape)
    mask[int(lower_cutoff_ratio * len(signal)):int(higher_cutoff_ratio * len(signal))] = 0
    filtered_signal = transformed_signal * mask
    denoised_signal = np.real(ifft(filtered_signal))
    return denoised_signal