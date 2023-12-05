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


class SEDDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        audio_dir,
        transformation,
        target_sample_rate,
        num_samples,
        label_column,
        device,
    ):
        """
        Args:
           index_path: the link to each audio
           idc: npy file, the number of samples in each class, computed in main
           config: the config.py module
           eval_model (bool): to indicate if the dataset is a testing dataset
        """
        self.annotations = pd.read_csv(annotations_file, sep="\t")
        self.audio_dir = audio_dir
        self.device = device
        self.waveform_transforms = False
        if transformation is not None:
            self.transformation = transformation.to(self.device)
        else:
            self.transformation = None
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.classes_num = config.classes_num
        print(self.classes_num)
        self.label_column = label_column
        total_size = len(self.annotations)

        logging.info("total dataset size: %d" % (total_size))
        logging.info("class num: %d" % (self.classes_num))

    def time_shifting(self, x):
        frame_num = len(x)
        shift_len = random.randint(0, self.shift_max - 1)
        new_sample = np.concatenate([x[shift_len:], x[:shift_len]], axis=0)
        return new_sample

    def crop_wav(self, x):
        crop_size = self.config.crop_size
        crop_pos = random.randint(0, len(x) - crop_size - 1)
        return x[crop_pos : crop_pos + crop_size]

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
        audio_sample_path = self._get_audio_sample_path(index)
        # print(audio_sample_path)
        label = self._get_audio_sample_label(index)
        # print(label)
        labels_list = []
        labels_list_int = []
        if self.label_column == "event_labels":
            labels_list = label.split(",")
            for lbl in labels_list:
                lbl_int = config.classes2id[lbl]
                labels_list_int.append(lbl_int)
        else:
            label_int = config.classes2id[label]
        # target = torch.zeros(self.classes_num)  # Initialize with zeros
        # target[label_int] = 1
        # audio_sample_path = self._get_audio_sample_path(index)

        # waveform, sr = sf.read(audio_sample_path)
        # waveform = torch.from_numpy(waveform)
        waveform, sr = torchaudio.load(audio_sample_path)
        waveform = self._resample_if_necessary(waveform, sr)
        waveform = self._mix_down_if_necessary(waveform)
        if waveform.shape[1] > self.num_samples:
            waveform = self._cut_if_necessary(waveform, 0.1, 10)

        waveform = self._right_pad_if_necessary(waveform)
        waveform = waveform.view(-1)

        if self.transformation is not None:
            waveform = self.transformation(waveform)

        labels = np.zeros(self.classes_num, dtype="f")
        if self.label_column == "event_labels":
            for lbl in labels_list:
                labels[config.classes2id[lbl]] = 1
        else:
            labels[config.classes2id[label]] = 1

        data_dict = {
            "audio_name": audio_sample_path,
            "waveform": waveform,
            "target": labels,
        }
        return data_dict

    def __len__(self):
        return len(self.annotations)

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        # print(path)
        return path

    def _get_audio_sample_label(self, index):
        column_index = self.annotations.columns.get_loc(self.label_column)
        return self.annotations.iloc[index, column_index]

    def _cut_if_necessary(self, signal, onset, offset):
        onset_frame = int(onset * self.target_sample_rate)
        offset_frame = int(offset * self.target_sample_rate)
        signal = signal[:, onset_frame:offset_frame]

        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
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
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
