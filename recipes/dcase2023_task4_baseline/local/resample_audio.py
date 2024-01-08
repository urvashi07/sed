import argparse
import glob
import os
from pathlib import Path

import librosa
import torch
import torchaudio
import multiprocessing as mp
import tqdm
from tqdm.contrib.concurrent import process_map
import yaml

def resample_data_generate_durations(config_data, test_only=False, evaluation=False):
    if not test_only:
        dsets = [
            "synth_folder",
            "synth_val_folder",
            "strong_folder",
            "weak_folder",
            "unlabeled_folder",
            "test_folder",
        ]
    elif not evaluation:
        dsets = ["test_folder"]
    else:
        dsets = ["eval_folder"]

    for dset in dsets:
        computed = resample_folder(
            os.path.join(config_data["prefix_folder"], config_data[dset + "_44k"]), 
            os.path.join(config_data["prefix_folder"], config_data[dset]),
            target_fs=config_data["fs"]
        )

    if not evaluation:
        for base_set in ["synth_val", "test"]:
            if not os.path.exists(config_data[base_set + "_dur"]) or computed:
                generate_tsv_wav_durations(
                    config_data[base_set + "_folder"], config_data[base_set + "_dur"]
                )



if __name__ == "__main__":

    conf_file_path = "/home/unegi2s/Documents/sed_github/sed/recipes/dcase2023_task4_baseline/confs/default.yaml"

    with open(conf_file_path, "r") as f:
        configs = yaml.safe_load(f)
    
    resample_data_generate_durations(configs["data"], test_only=False, evaluation=False)
