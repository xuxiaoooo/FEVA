import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from scipy.fft import fft, ifft

class FilterPreprocess:

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    @staticmethod
    def high_pass_filter(signal, cutoff_ratio=0.5):
        transformed_signal = fft(signal)
        mask = np.ones(signal.shape)
        mask[:int(cutoff_ratio * len(signal))] = 0
        filtered_signal = transformed_signal * mask
        denoised_signal = np.real(ifft(filtered_signal))
        return denoised_signal

    @staticmethod
    def low_pass_filter(signal, cutoff_ratio=0.5):
        transformed_signal = fft(signal)
        mask = np.ones(signal.shape)
        mask[int(cutoff_ratio * len(signal)):] = 0
        filtered_signal = transformed_signal * mask
        denoised_signal = np.real(ifft(filtered_signal))
        return denoised_signal

    @staticmethod
    def band_stop_filter(signal, lower_cutoff_ratio=0.25, higher_cutoff_ratio=0.75):
        transformed_signal = fft(signal)
        mask = np.ones(signal.shape)
        mask[int(lower_cutoff_ratio * len(signal)):int(higher_cutoff_ratio * len(signal))] = 0
        filtered_signal = transformed_signal * mask
        denoised_signal = np.real(ifft(filtered_signal))
        return denoised_signal

    def process_audio(self, waveform, sample_rate=None):
        # Ensure the waveform is of type float32
        waveform = waveform.float()

        # VAD (Voice Activity Detection)
        vad = torchaudio.transforms.Vad(sample_rate=sample_rate or self.sample_rate)
        voiced = vad(waveform)

        # Apply the three filters
        voiced_np = voiced.numpy()
        high_passed = self.high_pass_filter(voiced_np)
        low_passed = self.low_pass_filter(high_passed)
        band_stopped = self.band_stop_filter(low_passed)

        # Convert back to Tensor for further processing and ensure it's float32
        cleaned_waveform = torch.from_numpy(band_stopped).float()

        # Split the cleaned waveform into 3 segments
        num_frames = cleaned_waveform.shape[1]
        segment_size = num_frames // 3
        segments = [cleaned_waveform[:, i * segment_size:(i + 1) * segment_size] for i in range(3)]

        # Extract Mel spectrogram for each segment
        mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate or self.sample_rate, n_mels=128).to(dtype=torch.float32)
        mels = [mel_spectrogram(segment) for segment in segments]

        return mels[0], mels[1], mels[2]

    @classmethod
    def from_file(cls, audio_path, sample_rate=44100):
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        processor = cls(sample_rate=sr)
        return processor.process_audio(waveform, sample_rate=sr)

# 使用方法：
# processor = FilterPreprocess(sample_rate=44100)
# waveform, sr = torchaudio.load("path_to_your_audio_file.wav")
# mel1, mel2, mel3 = processor.process_audio(waveform, sample_rate=sr)

# 或者，如果你仍然想从文件加载：
# mel1, mel2, mel3 = FilterPreprocess.from_file("path_to_your_audio_file.wav")
