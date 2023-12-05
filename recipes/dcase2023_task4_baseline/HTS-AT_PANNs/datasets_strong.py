import pandas as pd
import torchaudio
from collections import OrderedDict
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torch.nn as nn
import logging
import config
import random
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


class SEDDataset_Strong(Dataset):
    def __init__(
        self,
        transformation,
        target_sample_rate,
        config,
        num_samples,
        list_audio_info,
        device,
    ):
        """
        Args:
           index_path: the link to each audio
           idc: npy file, the number of samples in each class, computed in main
           config: the config.py module
           eval_model (bool): to indicate if the dataset is a testing dataset
        """
        # self.annotations = pd.read_csv(annotations_file ,sep = "\t")
        # self.audio_dir = audio_dir
        self.device = device
        self.waveform_transforms = False
        if transformation is not None:
            self.transformation = transformation.to(self.device)
        else:
            self.transformation = None
        self.list_audio_info = list_audio_info
        # self.filenames = list(self.df["filename"].unique())
        self.target_sample_rate = target_sample_rate
        self.net_pooling = 1
        self.num_samples = num_samples
        self.configs = config
        self.classes_num = self.configs["classes_num"]
        # print(self.classes_num)
        # self.frame_hop = hop_length
        self.max_audio_duration = self.configs["feats"]["max_audio_duration"]
        self.sr = self.configs["feats"]["sample_rate"]
        self.hop_length = self.configs["feats"]["hop_length"]
        # total_size = len(self.annotations)
        self.audio_len = 10
        self.fmin = self.configs["feats"]["f_min"]
        self.fmax = self.configs["feats"]["f_max"]
        self.window_size = self.configs["feats"]["n_window"]
        self.mel_bins = self.configs["feats"]["n_mels"]
        n_frames = self.audio_len * self.num_samples
        self.n_frames = int(int((n_frames / self.hop_length)) / self.net_pooling)
        # logging.info("total dataset size: %d" %(total_size))
        logging.info("class num: %d" % (self.configs["classes_num"]))
        # self.event_dict = event_dict

        self.spectrogram_extractor = Spectrogram(
            n_fft=self.window_size,
            hop_length=self.hop_length,
            win_length=self.window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=self.sr,
            n_fft=self.window_size,
            n_mels=self.mel_bins,
            fmin=self.fmin,
            fmax=self.fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_mels=self.mel_bins,
            n_fft=self.window_size,
            win_length=self.window_size,
            hop_length=self.hop_length,
        )

    def time_shifting(self, x):
        frame_num = len(x)
        shift_len = random.randint(0, self.shift_max - 1)
        new_sample = np.concatenate([x[shift_len:], x[:shift_len]], axis=0)
        return new_sample

    def crop_wav(self, x):
        crop_size = self.configs["feats"]["crop_size"]
        crop_pos = random.randint(0, len(x) - crop_size - 1)
        return x[crop_pos : crop_pos + crop_size]

    def time_to_frame(self, time):
        return (self.sr * time) / self.hop_length

    """def _time_to_frame(self, time):
        #samples = time * self.num_samples
        #frame = ((samples) / self.frame_hop) - 1
        frame = time * config.htsat_spec_size * self.freq_ratio
        return np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames)"""

    def frame_to_time(self, frame):
        return (frame * self.hop_length) / self.sr

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
        filename = self.list_audio_info[index].filename
        filepath = self.list_audio_info[index].filepath
        onset_times = self.list_audio_info[index].onset_times
        offset_times = self.list_audio_info[index].offset_times
        class_labels = self.list_audio_info[index].class_labels
        waveform, sr = torchaudio.load(filepath)

        # tmp_data = np.array(self.event_dict[filename]).T
        frame_start = np.floor(self.time_to_frame(onset_times)).astype(int)
        frame_end = np.ceil(self.time_to_frame(offset_times)).astype(int)
        class_int = class_labels.astype(int)
        if waveform.shape[0] > 1:
            waveform = self._mix_down_if_necessary(waveform)
        if not waveform.shape[1] == self.target_sample_rate:
            waveform = self._resample_if_necessary(waveform, sr)

        waveform = self._cut_if_necessary(waveform, 0, 10)
        waveform = self._right_pad_if_necessary(waveform)
        waveform = waveform.view(-1)

        mbe = self.mel_spec(waveform).T
        labels_frames_arr = np.zeros((mbe.shape[0], self.classes_num))
        labels_arr = np.zeros(self.classes_num)

        for ind, val in enumerate(class_labels):
            labels_frames_arr[frame_start[ind] : frame_end[ind], val] = 1
            labels_arr[val] = 1
        data_dict = {
            "audio_name": filepath,
            "waveform": waveform,
            "target_frames": labels_frames_arr,
            "target_classes": labels_arr,
        }
        return data_dict

    def __len__(self):
        return len(self.list_audio_info)

    """def _get_onset_time(self, index):
        column_index = self.annotations.columns.get_loc(self.onset_column)
        return self.annotations.iloc[index, column_index]
    
    def _get_offset_time(self, index):
        column_index = self.annotations.columns.get_loc(self.offset_column)
        return self.annotations.iloc[index, column_index]"""

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        # print(path)
        return path

    def _cut_if_necessary(self, signal, onset, offset):
        # onset_frame = int(onset * self.target_sample_rate)
        # offset_frame = int(offset * self.target_sample_rate *self.max_audio_duration)
        # signal = signal[:, onset_frame:offset_frame]
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
        if not sr == self.target_sample_rate:
            signal1 = torchaudio.functional.resample(
                signal, orig_freq=sr, new_freq=self.target_sample_rate
            )
            return signal1
        else:
            return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
