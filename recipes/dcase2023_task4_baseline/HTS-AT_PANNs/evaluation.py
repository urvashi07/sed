import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
import torchaudio
from panns_models import *

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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, sampler, Subset, ConcatDataset

from prepare_data import prepare_all_data

import config
from sed_model import SEDWrapper, Ensemble_SEDWrapper
from models import Cnn14_DecisionLevelMax

from model.htsat import HTSAT_Swin_Transformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
import warnings
from collections import OrderedDict
import argparse

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "desed_task/dataio/")
)
from sampler import ConcatDatasetBatchSampler
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings("ignore")

from datasets import SEDDataset
from datasets_strong import SEDDataset_Strong

# from typing import List
from pathlib import Path
import numpy as np
from prepare_data import prepare_all_data

from create_hdf_file import create_hdf_file
import h5py

def is_hdf5_empty(file_path):
    try:
        with h5py.File(file_path, 'r') as file:
            return len(file.keys()) == 0
    except OSError:
        return True
    
if __name__ == "__main__":
    start_time = time.time()
    print(
        f"Begin time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str, help="Model to be used: panns, hts-at", default="panns"
    )
    args = parser.parse_args()
    print(args.model)

    args.model = args.model.lower()
    config.model = args.model
    log_dir = ""
    if args.model == "panns":
        log_dir = os.path.join("./logs", "panns_all_data")
    elif args.model == "hts-at" or args.model == "htsat":
        log_dir = os.path.join("./logs", "hts-at_all_data")
    else:
        print(args.model + " not defined currently. Only PANNs and HTS-AT defined.")
        sys.exit()

    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    conf_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "confs/default.yaml",
    )

    with open(conf_file_path, "r") as f:
        configs = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)
    pl.utilities.seed.seed_everything(seed=configs["training"]["random_seed"])
    logging.info(
        f"Begin time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )
    logging.info("Epochs: " + str(configs["training"]["max_epoch"]))
    logging.info("***************Model****************")
    logging.info(args.model)
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
    # BATCH_SIZE = 8

    # frame_length_in_seconds
    frame_length_sec = HOP_LENGTH / SAMPLE_RATE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("device: " + device)

    hdf_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), configs["hdf_file"])

    if is_hdf5_empty(hdf_file_path):
        data = prepare_all_data(configs["data"])
        create_hdf_file(data, hdf_file_path, sr=SAMPLE_RATE)

    h5py_file = h5py.File(hdf_file_path, "r")

    train_data = h5py_file["train"]
    eval_data = h5py_file["eval"]
    test_data = h5py_file["test"]

    test_dataset = SEDDataset_Strong(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        data=test_data["strong"],
        config=configs,
        device=device,
    )
    
    test_sampler = None
    test_dataloader = DataLoader(
            dataset=test_dataset,
            num_workers=configs["training"]["num_workers"],
            batch_size=configs["training"]["batch_size"],
            shuffle=False,
            sampler=test_sampler,
        )
    
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name=args.model)

    checkpoint_callback = ModelCheckpoint(
        monitor="mAP",
        dirpath="checkpoints/l-{epoch:d}-{mAP:.3f}",
        # filename='l-{epoch:d}-{mAP:.3f}',
        save_top_k=2,
        mode="max",
    )
    early_stop = EarlyStopping(
                monitor="mAP",
                patience=configs["training"]["early_stop_patience"],
                verbose=True,
                mode="max",
            )
    
    trainer = pl.Trainer(
        deterministic=False,
        #accelerator="cpu",  # For running locally,
        accelerator="gpu",
        # gpus=None,  # For running locally,
        gpus=[0],
        max_epochs=configs["training"]["max_epoch"],
        auto_lr_find=False,
        sync_batchnorm=True,
        num_sanity_val_steps=0,
        # resume_from_checkpoint = config.resume_checkpoint,
        gradient_clip_val=1.0,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop, TQDMProgressBar(refresh_rate=1)]
    )

    ckpt_epoch = str(configs["training"]["ckpt_epoch"])
    if args.model == "hts-at" or args.model == "htsat":
        sed_model = HTSAT_Swin_Transformer(
            spec_size=configs["hts-at"]["htsat_spec_size"],
            patch_size=configs["hts-at"]["htsat_patch_size"],
            in_chans=1,
            num_classes=configs["classes_num"],
            window_size=configs["hts-at"]["htsat_window_size"],
            config=config,
            depths=configs["hts-at"]["htsat_depth"],
            embed_dim=configs["hts-at"]["htsat_dim"],
            patch_stride=tuple(configs["hts-at"]["htsat_stride"]),
            num_heads=configs["hts-at"]["htsat_num_head"],
        )

        ckpt_path = os.path.join(configs["data"]["prefix_folder"], configs["data"]["ckpt_htsat_" + ckpt_epoch])

    elif args.model == "panns":
        # model
        model_config = {
            "sample_rate": SAMPLE_RATE,
            "window_size": WIN_LENGTH,
            "hop_size": HOP_LENGTH,
            "mel_bins": 64,
            "fmin": F_MIN,
            "fmax": F_MAX,
            "classes_num": 10,
        }
        sed_model = PANNsCNN14Att(**model_config)
        # weights = torch.load("Cnn14_DecisionLevelAtt_mAP0.425.pth", map_location = "cpu")
        # Fixed in V3
        # model.load_state_dict(weights["model"])
        sed_model.att_block = AttBlock(2048, 10, activation="sigmoid")

        ckpt_path = os.path.join(configs["data"]["prefix_folder"], configs["data"]["ckpt_panns_" + ckpt_epoch])

    # model = SEDWrapper(sed_model=sed_model, config=config, df_eval = pd.concat([df_train_strong, df_train_synth]).iloc[list_val_indices],
    #                   df_test = pd.concat([df_eval_strong, df_eval_strong]), prefix_folder = configs["data"]["prefix_folder"])
    model = SEDWrapper(
        sed_model=sed_model,
        config=config,
        prefix_folder=configs["data"]["prefix_folder"],
    )

    ckpt = torch.load(ckpt_path, map_location = "cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    trainer.test(model, test_dataloader)
    h5py_file.close()
    end_time = time.time()
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

    logging.info(
        f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
    )

    logging.info(f"Total time: {end_time - start_time}")
