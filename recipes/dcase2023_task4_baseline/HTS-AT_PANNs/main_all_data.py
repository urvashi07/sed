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

from utils import (
    create_folder,
    dump_config,
    process_idc,
    prepprocess_audio,
    init_hier_head,
)

import config
from sed_model import SEDWrapper, Ensemble_SEDWrapper
from models import Cnn14_DecisionLevelMax

# from data_generator import SEDDataset, DESED_Dataset, ESC_Dataset, SCV2_Dataset

from model.htsat import HTSAT_Swin_Transformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
from collections import OrderedDict
import argparse

sys.path.append(
    "/home/unegi/Documents/dcase2023_task4_gitlab/DESED_task/desed_task/dataio/"
)
from sampler import ConcatDatasetBatchSampler

warnings.filterwarnings("ignore")

from datasets import SEDDataset
from datasets_strong import SEDDataset_Strong

# from typing import List
from pathlib import Path
import numpy as np
from prepare_data import prepare_all_data

print(torch.cuda.is_available())


def custom_collate(batch):
    audio_names = [item["audio_name"] for item in batch]
    waveforms = [item["waveform"] for item in batch]
    targets = [item["target"] for item in batch]

    return {
        "audio_names": audio_names,
        "waveforms": waveforms,
        "targets": targets,
    }


# Create a custom DataLoader that uses the ConcatDatasetBatchSampler


class data_prep(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, test_dataset, 
                 train_batch_sampler):#,
                 #val_batch_sampler):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.train_batch_sampler = train_batch_sampler
        #self.val_batch_sampler = val_batch_sampler
        # self.device_num = device_num

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            num_workers=configs["training"]["num_workers"],
            shuffle=False,
            batch_sampler=self.train_batch_sampler,
            collate_fn=custom_collate,
        )
        return train_loader

    def val_dataloader(self):
        eval_loader = DataLoader(
            dataset=self.eval_dataset,
            num_workers=configs["training"]["num_workers"],
            shuffle=False,
            sampler=None,
            batch_size=4,
            #sampler=self.val_batch_sampler,
            #collate_fn=custom_collate,
        )
        return eval_loader

    def test_dataloader(self):
        test_sampler = None
        test_loader = DataLoader(
            dataset=self.test_dataset,
            num_workers=configs["training"]["num_workers"],
            batch_size=configs["training"]["batch_size"],
            shuffle=False,
            sampler=test_sampler,
        )
        return test_loader


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

    log_dir = ""
    if args.model == "panns":
        log_dir = os.path.join("./logs", "panns_strong")
    elif args.model == "hts-at" or args.model == "htsat":
        log_dir = os.path.join("./logs", "hts-at_strong")
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

    data = prepare_all_data(configs["data"])

    train_dataset_strong = SEDDataset_Strong(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        list_audio_info=data["train"]["strong"],
        config=configs,
        device=device,
    )

    train_dataset_synth = SEDDataset_Strong(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        list_audio_info=data["train"]["synth"],
        config=configs,
        device=device,
    )

    train_dataset_weak = SEDDataset(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        list_audio_info=data["train"]["weak"],
        config=configs,
        device=device,
    )

    train_dataset_strong_synth = torch.utils.data.ConcatDataset(
        [train_dataset_strong, train_dataset_synth]
    )
    print(len(train_dataset_strong_synth))
    tot_train_data = [train_dataset_strong_synth, train_dataset_weak]
    print(len(train_dataset_weak))
    train_dataset = torch.utils.data.ConcatDataset(tot_train_data)
    batch_sizes = [4, 2]
    samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
    batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

    percentage_train = int(np.ceil(0.95 * len(train_dataset)))
    list_train_indices = [num for num in range(percentage_train)]
    list_val_indices = [num for num in range(percentage_train, len(train_dataset))]

    # val_dataset = Subset(train_dataset, list_val_indices)
    # train_dataset = Subset(train_dataset, list_train_indices)
    # val_dataset = Subset(train_dataset, list_val_indices)

    eval_dataset_synth = SEDDataset_Strong(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        list_audio_info=data["eval"]["synth"],
        config=configs,
        device=device,
    )

    eval_dataset_weak = SEDDataset(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        list_audio_info=data["eval"]["weak"],
        config=configs,
        device=device,
    )

    test_dataset = SEDDataset_Strong(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        list_audio_info=data["eval"]["strong"],
        config=configs,
        device=device,
    )

    # eval_dataset = ConcatDataset([eval_dataset_strong, eval_dataset_synth])
    # eval_dataset = eval_dataset_strong

    # print("***********************************")
    # print(train_dataset[0]["waveform"].shape)
    # print(val_dataset[0]["waveform"].shape)

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name=args.model)
    batch_sizes_val = [6,6]
    total_eval_data = [eval_dataset_synth, eval_dataset_weak]
    val_data = torch.utils.data.ConcatDataset(total_eval_data)
    val_data = torch.utils.data.Subset(val_data, np.arange(8))
    #sampler_eval = [torch.utils.data.RandomSampler(x) for x in total_eval_data]
    #batch_sampler_val = ConcatDatasetBatchSampler(sampler_eval, batch_sizes_val)


    """if configs["training"]["reduce_dataset_size"]:
        train_dataset = Subset(train_dataset, np.arange(10))
        val_dataset = Subset(eval_dataset, np.arange(5))
        eval_dataset = Subset(test_dataset, np.arange(3))"""
    test_dataset = Subset(test_dataset, np.arange(3))
    # batch_sizes_all_data = [1, 1]

    # print("***********************************")
    # print(type(eval_dataset))

    sed_data = data_prep(train_dataset, val_data, 
                         test_dataset, batch_sampler)
                         #batch_sampler_val)
    for batch in sed_data.train_dataloader():
        print(batch)
        print(type(batch))
        break
    print("..............................")

    for batch in sed_data.val_dataloader():
        print(batch)
        print(type(batch))
        break

    checkpoint_callback = ModelCheckpoint(
        monitor="mAP",
        dirpath="checkpoints/l-{epoch:d}-{mAP:.3f}",
        # filename='l-{epoch:d}-{mAP:.3f}',
        save_top_k=2,
        mode="max",
    )

    trainer = pl.Trainer(
        deterministic=False,
        accelerator="cpu",  # For running locally,
        # accelerator="gpu",
        gpus=None,  # For running locally,
        # gpus=[0],
        max_epochs=1,
        # max_epochs=configs["training"]["max_epoch"],
        auto_lr_find=False,
        sync_batchnorm=True,
        num_sanity_val_steps=0,
        # resume_from_checkpoint = config.resume_checkpoint,
        gradient_clip_val=1.0,
        logger=tb_logger,
    )

    pretrain_path = ""

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

        pretrain_path = os.path.join(
            configs["data"]["prefix_folder"], configs["swin_pretrain_path"]
        )

    elif args.model == "panns":
        # model
        model_config = {
            "sample_rate": SAMPLE_RATE,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 10,
        }
        sed_model = PANNsCNN14Att(**model_config)
        # weights = torch.load("Cnn14_DecisionLevelAtt_mAP0.425.pth", map_location = "cpu")
        # Fixed in V3
        # model.load_state_dict(weights["model"])
        sed_model.att_block = AttBlock(2048, 10, activation="sigmoid")

        pretrain_path = os.path.join(
            configs["data"]["prefix_folder"], configs["panns_pretrain_path"]
        )

    # model = SEDWrapper(sed_model=sed_model, config=config, df_eval = pd.concat([df_train_strong, df_train_synth]).iloc[list_val_indices],
    #                   df_test = pd.concat([df_eval_strong, df_eval_strong]), prefix_folder = configs["data"]["prefix_folder"])
    model = SEDWrapper(
        sed_model=sed_model,
        config=config,
        prefix_folder=configs["data"]["prefix_folder"],
    )

    model.learning_rate = LEARNING_RATE

    trainer.tune(model, datamodule=sed_data)
    sed_data.setup("fit")
    suggested_lr = (
        model.learning_rate
    )  # Access the suggested learning rate from the model
    print(f"Suggested learning rate: {model.learning_rate:.2e}")

    if pretrain_path is not None:  # train with pretrained model
        if args.model == "hts-at" or args.model == "htsat":
            ckpt = torch.load(pretrain_path, map_location="cpu")
            ckpt["state_dict"].pop("sed_model.head.weight")
            ckpt["state_dict"].pop("sed_model.head.bias")
            # finetune on the esc and spv2 dataset
            ckpt["state_dict"].pop("sed_model.tscam_conv.weight")
            ckpt["state_dict"].pop("sed_model.tscam_conv.bias")
            model.load_state_dict(ckpt["state_dict"], strict=False)

        elif args.model == "panns":
            ckpt = torch.load(pretrain_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)

    trainer.fit(model, sed_data.train_dataloader(), sed_data.val_dataloader())

    # best_model = SEDWrapper.load_from_checkpoint(checkpoint_callback.best_model_path)

    """prediction_df = best_model.prediction(test_df=df_eval_strong,
                           test_audio=os.path.join(configs["data"]["prefix_folder"], configs["data"]["val_folder"]),
                           threshold=0.5, 
                           SR= SAMPLE_RATE)"""

    """trainer = pl.Trainer(
        deterministic=False,
        gpus = 0, 
        max_epochs = config.max_epoch,   
        sync_batchnorm = True,
        num_sanity_val_steps = 0,
        # resume_from_checkpoint = config.resume_checkpoint,
        gradient_clip_val=1.0,
        logger=tb_logger,

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
        dataset = eval_dataset
    )
    if config.resume_checkpoint is not None:
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    #else:
        #best_path = trainer.checkpoint_callback.best_model_path
        #print(f"best model: {best_path}")
        #test_state_dict = torch.load(best_path)["state_dict"]
        #model.load_state_dict(test_state_dict)"""

    trainer.test(model, sed_data.test_dataloader(), ckpt_path="best")

    end_time = time.time()
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

    logging.info(
        f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
    )

    logging.info(f"Total time: {end_time - start_time}")
