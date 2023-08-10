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
from torch.utils.data import Dataset, DataLoader, sampler, Subset

from utils import create_folder, dump_config, process_idc, prepprocess_audio, init_hier_head

import config
from sed_model import SEDWrapper, Ensemble_SEDWrapper
from models import Cnn14_DecisionLevelMax
#from data_generator import SEDDataset, DESED_Dataset, ESC_Dataset, SCV2_Dataset

from model.htsat import HTSAT_Swin_Transformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
from collections import OrderedDict
import argparse

warnings.filterwarnings("ignore")

from datasets import SEDDataset

print(torch.cuda.is_available())
#sys.exit()

class data_prep(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, test_dataset):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
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
    
    def val_dataloader(self):
        eval_sampler = None
        eval_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size,
            shuffle = False,
            sampler = eval_sampler
        )
        return eval_loader
    
    def test_dataloader(self):
        test_sampler = None
        test_loader = DataLoader(
            dataset = self.test_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size,
            shuffle = False,
            sampler = test_sampler
        )
        return test_loader

if __name__ == "__main__":
    start_time = time.time()
    print(f"Begin time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model to be used: panns, hts-at', default = "panns")
    args = parser.parse_args()
    print(args.model)

    args.model = args.model.lower()

    log_dir = ""
    if args.model == "panns":
        log_dir = os.path.join("./logs", "panns_updated")
    elif args.model == "hts-at" or args.model == "htsat":
        log_dir = os.path.join("./logs", "hts-at_updated")
    else:
        print(args.model + " not defined currently. Only PANNs and HTS-AT defined.")
        sys.exit()

    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logging.basicConfig(level=logging.INFO) 
    pl.utilities.seed.seed_everything(seed = config.random_seed)
    logging.info(f"Begin time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    logging.info("Epochs: " + str(config.max_epoch))
    logging.info("***************Model****************")
    logging.info(args.model)

    with open("../confs/default.yaml", "r") as f:
        configs = yaml.safe_load(f)
    
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
    BATCH_SIZE = 8

    #frame_length_in_seconds
    frame_length_sec = HOP_LENGTH / SAMPLE_RATE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("device: " +device)

    train_dataset = SEDDataset(annotations_file = "../" + configs["data"]["weak_tsv"], 
                                          audio_dir = "../" + configs["data"]["weak_folder"], 
                                          transformation = None, 
                                          target_sample_rate = SAMPLE_RATE,
                                          num_samples = NUM_SAMPLES,
                                          label_column = "event_labels",
                                          device = device)

    percentage_train = int(np.ceil(0.95*len(train_dataset)))
    list_train_indices = [num for num in range(percentage_train)]
    list_val_indices = [num for num in range(percentage_train, len(train_dataset))]

    val_dataset = Subset(train_dataset, list_val_indices)
    train_dataset = Subset(train_dataset, list_train_indices)
    #val_dataset = Subset(train_dataset, list_val_indices)


    eval_dataset = SEDDataset(annotations_file = "../" +configs["data"]["val_tsv"],
                                          audio_dir = "../" +configs["data"]["val_folder"],
                                          transformation = None, 
                                          target_sample_rate = SAMPLE_RATE,
                                          num_samples = NUM_SAMPLES,
                                          label_column = "event_label",
                                          device = device)

    #print("***********************************")
    #print(train_dataset[0]["waveform"].shape)
    #print(val_dataset[0]["waveform"].shape)

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name=args.model)
    
    if config.reduce_dataset_size:
        train_dataset = Subset(train_dataset, np.arange(20))
        val_dataset = Subset(val_dataset, np.arange(10))
        eval_dataset = Subset(eval_dataset, np.arange(5))

    #print("***********************************")
    #print(type(eval_dataset))

    audioset_data = data_prep(train_dataset, val_dataset, eval_dataset)

    checkpoint_callback = ModelCheckpoint(
            monitor = "mAP",
            dirpath = "checkpoints/l-{epoch:d}-{mAP:.3f}",
            #filename='l-{epoch:d}-{mAP:.3f}',
            save_top_k = 2,
            mode = "max"
        )

    trainer = pl.Trainer(
        deterministic=False,
        gpus = None, 
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        num_sanity_val_steps = 0,
        # resume_from_checkpoint = config.resume_checkpoint,
        gradient_clip_val=1.0,
        logger=tb_logger,
    )

    

    pretrain_path = ""

    if args.model == "hts-at" or args.model == "htsat":
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
    

        pretrain_path = config.swin_pretrain_path

    elif args.model == "panns":
        # model
        model_config = {
            "sample_rate": 16000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 10
                }
        model_config["classes_num"] = 10
        sed_model = PANNsCNN14Att(**model_config)
        #weights = torch.load("Cnn14_DecisionLevelAtt_mAP0.425.pth", map_location = "cpu")
        # Fixed in V3
        #model.load_state_dict(weights["model"])
        sed_model.att_block = AttBlock(2048, 10, activation='sigmoid')

        pretrain_path = config.panns_pretrain_path

    model = SEDWrapper(
        sed_model = sed_model, 
        config = config,
    )
    
    if pretrain_path is not None: # train with pretrained model
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

    trainer.fit(model, audioset_data.train_dataloader(), audioset_data.val_dataloader())


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

    trainer.test(model, audioset_data.test_dataloader(), ckpt_path = "best")

    end_time = time.time()
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

    logging.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

    logging.info(f"Total time: {end_time - start_time}")