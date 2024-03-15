import torch
import torchaudio
from audiomentations import PitchShift, AddGaussianNoise
import numpy as np
import random
from torch.utils.data import Dataset

def shift_pitch(waveform, sample_rate):
    pitch_shifter = PitchShift(min_semitones=-0.5, max_semitones=0.5, p=1.0)
    pitch_shifted_audio = pitch_shifter(samples=np.array(waveform), sample_rate=sample_rate)
    return torch.from_numpy(pitch_shifted_audio)

def add_gaussian_noise(waveform, sample_rate):
    transform = AddGaussianNoise(
    min_amplitude=0.001,
    max_amplitude=0.015,
    p=1.0
    )
    augmented_audio = transform(waveform.numpy(), sample_rate=sample_rate)
    return torch.from_numpy(augmented_audio)

def augment(waveform, sample_rate):
    random_number = random.randint(1, 4)
    for i in range(random_number):
        random_augment = random.randint(1, 2)
        if random_augment == 1:
            waveform = shift_pitch(waveform, sample_rate=sample_rate)
        elif random_augment == 2:
            waveform = add_gaussian_noise(waveform, sample_rate=sample_rate)
    return waveform

class AugmentedDataset(Dataset):
    def __init__(self, augmented_audio_dict):
        self.augmented_audio_dict = augmented_audio_dict

    def __getitem__(self, index):
        item = self.augmented_audio_dict[index]
        waveform = item['waveform']
        target = item['target']
        audio_name = item['audio_name']

        return {'waveform': waveform, 
                'target': target, 
                'audio_name': audio_name}

    def __len__(self):
        return len(self.augmented_audio_dict)


def augment_audio_files(dataset):
    augmented_audio_dict = {}
    len_dataset = len(dataset)
    data_augment_files = int(len_dataset/3)
    for idx in range(data_augment_files):
        random_index = random.randint(0, len_dataset - 1)
        random_element = dataset[random_index]
        augmented_audio = augment(random_element["waveform"], sample_rate=16000)
        augmented_audio_dict[idx] = {"waveform": augmented_audio,
                                      "target": random_element["target"], 
                                      "audio_name": random_element["audio_name"]}
    return AugmentedDataset(augmented_audio_dict)
