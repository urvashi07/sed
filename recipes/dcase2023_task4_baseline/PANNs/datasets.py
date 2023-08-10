import pandas as pd
import torchaudio
from collections import OrderedDict
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torch.nn as nn

class PANNsDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, 
                 transformation, target_sample_rate, 
                 num_samples, device):
        self.annotations = pd.read_csv(annotations_file ,sep = "\t")
        self.audio_dir = audio_dir
        self.device = device
        if transformation is not None:
            self.transformation = transformation.to(self.device)
        else:
            self.transformation = None
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        self.classes2id = OrderedDict(
        {
        "Alarm_bell_ringing": 0,
        "Blender": 1,
        "Cat": 2,
        "Dishes": 3,
        "Dog": 4,
        "Electric_shaver_toothbrush": 5,
        "Frying": 6,
        "Running_water": 7,
        "Speech": 8,
        "Vacuum_cleaner": 9,
        }
        )

        self.id2classes = {value: key for key, value in self.classes2id.items()}
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        audio_sample_path = self._get_audio_sample_path(idx)
        label = self._get_audio_sample_label(idx)
        label_int = self.classes2id[label]
        onset = self._get_audio_onset_time(idx)
        offset = self._get_audio_offset_time(idx)

        #waveform = torch.from_numpy(waveform)
        waveform, sr = torchaudio.load(audio_sample_path)

        waveform = self._resample_if_necessary(waveform, sr)
        
        waveform = self._mix_down_if_necessary(waveform)

        if waveform.shape[1] > self.num_samples:
            waveform = self._cut_if_necessary(waveform, onset, offset)
           
        waveform = self._right_pad_if_necessary(waveform)
        #make the signal.shape = (1,...)
        waveform = waveform.view(-1)


        if self.transformation is not None:
            waveform = self.transformation(waveform)

        labels = np.zeros(len(self.classes2id), dtype="f")
        labels[self.classes2id[label]] = 1
        
          # Convert numpy array to Tensor
        #waveform = torch.mean(waveform, dim=0, keepdim=True)

        #waveform_np = waveform.numpy()
        #print(waveform_np.shape)
        #print(waveform_np.size)
        

        return {"waveform": waveform, "targets": labels}
    
    def _cut_if_necessary(self, signal, onset, offset):

        onset_frame = int(onset * self.target_sample_rate)
        offset_frame = int(offset * self.target_sample_rate)
        signal = signal[:, onset_frame:offset_frame]

        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
        if not sr == self.target_sample_rate:
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim = 0, keepdim = True)
        return signal

    
    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        #print(path)
        return path
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 3]
    
    def _get_audio_onset_time(self, index):
        return self.annotations.iloc[index, 1]
    
    def _get_audio_offset_time(self, index):
        return self.annotations.iloc[index, 2]