import torch
import torch.nn as nn
import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
import torchaudio

import os
from re import A, S
import sys
import librosa
import numpy as np
import argparse
import h5py
import math
import time
import logging
import pickle
import random
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torch.utils.data.distributed import DistributedSampler

from utils import create_folder, dump_config, process_idc, prepprocess_audio, init_hier_head

import config
from sed_model import SEDWrapper, Ensemble_SEDWrapper
from models import Cnn14_DecisionLevelMax
from data_generator import SEDDataset, DESED_Dataset, ESC_Dataset, SCV2_Dataset


from model.htsat import HTSAT_Swin_Transformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings



warnings.filterwarnings("ignore")

from torch.utils.data import Dataset

with open("../confs/default.yaml", "r") as f:
        configs = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO) 
pl.utilities.seed.seed_everything(seed = config.random_seed)

from collections import OrderedDict


classes2id = OrderedDict(
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

id2classes = {value: key for key, value in classes2id.items()}

class SEDDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, 
                 transformation, target_sample_rate,
                 num_samples, device):
        """
        Args:
           index_path: the link to each audio
           idc: npy file, the number of samples in each class, computed in main
           config: the config.py module 
           eval_model (bool): to indicate if the dataset is a testing dataset
        """
        self.annotations = pd.read_csv(annotations_file ,sep = "\t")
        self.audio_dir = audio_dir
        self.device = device
        self.waveform_transforms = False
        #self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.classes_num = len(classes2id)
        total_size = len(self.annotations)
        
        logging.info("total dataset size: %d" %(total_size))
        logging.info("class num: %d" %(self.classes_num))

    def time_shifting(self, x):
        frame_num = len(x)
        shift_len = random.randint(0, self.shift_max - 1)
        new_sample = np.concatenate([x[shift_len:], x[:shift_len]], axis = 0)
        return new_sample 

    def crop_wav(self, x):
        crop_size = self.config.crop_size
        crop_pos = random.randint(0, len(x) - crop_size - 1)
        return x[crop_pos:crop_pos + crop_size]

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
        label = self._get_audio_sample_label(index)
        label_int = classes2id[label]
        audio_sample_path = self._get_audio_sample_path(index)

        #waveform, sr = sf.read(audio_sample_path)
        #waveform = torch.from_numpy(waveform)
        waveform, sr = torchaudio.load(audio_sample_path)
        waveform = self._resample_if_necessary(waveform, sr)
        waveform = self._mix_down_if_necessary(waveform)
        #if waveform.shape[1] > self.num_samples:
            #waveform = self._cut_if_necessary(waveform, onset, offset)
           
        waveform = self._right_pad_if_necessary(waveform)
        waveform = waveform.view(-1)
        
        data_dict = {
                "audio_name": audio_sample_path,
                "waveform": waveform,
                "target": label_int
            }
        return data_dict

    def __len__(self):
        return len(self.annotations)
    
    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        #print(path)
        return path
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 3]
    
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

SAMPLE_RATE = configs["data"]["fs"]
N_FFT = configs["feats"]["n_window"]
WIN_LENGTH = configs["feats"]["n_window"]
HOP_LENGTH = configs["feats"]["hop_length"]
F_MIN = configs["feats"]["f_min"]
F_MAX = configs["feats"]["f_max"]
N_MELS = configs["feats"]["n_mels"]
WINDOW_FN = torch.hamming_window
WKWARGS = {"periodic": False}
POWER = 1
NUM_SAMPLES = SAMPLE_RATE

LEARNING_RATE = configs["opt"]["lr"]
epochs = 5
BATCH_SIZE = 8

    #frame_length_in_seconds
frame_length_sec = HOP_LENGTH / SAMPLE_RATE
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = SEDDataset(annotations_file = "../" + configs["data"]["synth_tsv"], 
                                          audio_dir = "../" + configs["data"]["synth_folder"], 
                                          transformation = None, 
                                          target_sample_rate = SAMPLE_RATE,
                                          num_samples = NUM_SAMPLES,
                                          device = device)

class data_prep(pl.LightningDataModule):
    def __init__(self, train_dataset):
        super().__init__()
        self.train_dataset = train_dataset
        #self.eval_dataset = eval_dataset
        #self.device_num = device_num

    def train_dataloader(self):
        train_sampler = None
        train_loader = DataLoader(
            dataset = self.train_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size,
            shuffle = False,
            sampler = train_sampler
        )
        return train_loader
    
audioset_data = data_prep(train_dataset)

checkpoint_callback = ModelCheckpoint(
            monitor = "mAP",
            filename='l-{epoch:d}-{mAP:.3f}-{mAUC:.3f}',
            save_top_k = 20,
            mode = "max"
        )

trainer = pl.Trainer(
        deterministic=False,
        max_epochs = 1,
        auto_lr_find = True,    
        sync_batchnorm = True,
        num_sanity_val_steps = 0,
        # resume_from_checkpoint = config.resume_checkpoint,
        gradient_clip_val=1.0
    )

sed_model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config = config,
        depths = config.htsat_depth,
        embed_dim = config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )


model = SEDWrapper(
        sed_model = sed_model, 
        config = config,
        dataset = train_dataset
    )

trainer.fit(model, audioset_data.train_dataloader())
