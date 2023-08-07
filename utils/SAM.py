import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from filters import high_pass_filter, low_pass_filter, band_stop_filter

# The size of the Mel spectrogram
SPECTROGRAM_SIZE = (128, 128)
# The size of the patches
PATCH_SIZE = (4, 4)

# SegmentedAudioprocessingModule
class SAM(Dataset):
    def __init__(self, file_list, label_list):
        self.file_list = file_list
        self.label_list = label_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.file_list[idx])

        # Split the waveform into N segments
        segments = torch.split(waveform, waveform.size(0) // 5, dim=0)
        processed_segments = []
        for segment in segments:
            segment_np = segment.numpy()
            high_pass_segment = high_pass_filter(segment_np)
            low_pass_segment = low_pass_filter(segment_np)
            band_stop_segment = band_stop_filter(segment_np)

            # Create the Mel spectrograms
            mel_transform = T.MelSpectrogram(sample_rate, n_mels=SPECTROGRAM_SIZE[0])
            high_pass_mel_spectrogram = mel_transform(torch.from_numpy(high_pass_segment))
            low_pass_mel_spectrogram = mel_transform(torch.from_numpy(low_pass_segment))
            band_stop_mel_spectrogram = mel_transform(torch.from_numpy(band_stop_segment))

            # Stack the Mel spectrograms
            stacked_spectrograms = torch.stack([high_pass_mel_spectrogram, low_pass_mel_spectrogram, band_stop_mel_spectrogram])

            # Resize and normalize the spectrograms to ensure they have the same size
            resize_transform = T.ResizeSpectrogram(SPECTROGRAM_SIZE)
            stacked_spectrograms = resize_transform(stacked_spectrograms)

            # Split into patches and reshape into 1D
            patches = torch.nn.functional.unfold(stacked_spectrograms.unsqueeze(0), PATCH_SIZE).squeeze(0)
            patches = patches.view(-1, patches.shape[-1])  # Flatten to 1D
            patches = patches.permute(1, 0)  # Transpose so that each row is a patch

            processed_segments.append(patches)

        return processed_segments, self.label_list[idx]
