import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import os
import numpy as np

class UnlabelledDataset(Dataset):
    def __init__(
        self,
        config,
        num_samples,
        dirpath,
    ):
        self.dirpath = dirpath
        self.files = os.listdir(self.dirpath)
        
        self.num_samples = num_samples
        self.configs = config
        self.classes_num = self.configs["classes_num"]
        self.max_audio_duration = self.configs["feats"]["max_audio_duration"]
        self.sr = self.configs["feats"]["sample_rate"]
        self.hop_length = self.configs["feats"]["hop_length"]

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "hdf5_path": str,
            "index_in_hdf5": int,
            "audio_name": str,
            "waveform": (clip_samples,),
            "target": (classes_num,)
        }
        """
        filepath = os.path.join(self.dirpath, self.files[index])
        waveform, sr = torchaudio.load(filepath)
        waveform = self._resample_if_necessary(waveform, sr)
        waveform = self._mix_down_if_necessary(waveform)
        waveform = self._cut_if_necessary(waveform)
        waveform = self._right_pad_if_necessary(waveform)

        waveform = waveform.view(-1)
        return {"waveform": waveform,
                "audio_name": filepath}
        
    def __len__(self):
        return len(self.files)
    
    def _cut_if_necessary(self, signal):
        max_length = self.sr * self.max_audio_duration
        if signal.shape[1] > max_length:
            signal = signal[:, : (self.sr * self.max_audio_duration)]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        max_length = self.sr * self.max_audio_duration
        
        if length_signal < max_length:
            num_missing_samples = max_length - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        # resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
        if not sr == self.sr:
            signal1 = torchaudio.functional.resample(
                signal, orig_freq=sr, new_freq=self.target_sample_rate
            )
            return signal1
        else:
            return signal
        
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim = 0, keepdim = True)
        return signal
        

class Unlabelled2Weak(Dataset):
    def __init__(
        self,
        data
    ):
        self.data

    def __getitem__(self, index):
        return {"waveform": self.data[index]["waveform"],
                "target": self.data[index]["predictions"],
                "audio_name": self.data[index]["audio_name"]}

    def __len__(self):
        return len(self.data)