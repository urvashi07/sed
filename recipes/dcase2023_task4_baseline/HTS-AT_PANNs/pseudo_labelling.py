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
from dataset_unlabelled import UnlabelledDataset, Unlabelled2Weak
from AudioClassification import AudioClassification

# from typing import List
from pathlib import Path
import numpy as np
from prepare_data import prepare_all_data

from create_hdf_file import create_hdf_file
import h5py

from augment import augment_audio_files

sys.path.append(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            )
from desed_task.utils.schedulers import ExponentialWarmup

def is_hdf5_empty(file_path):
    try:
        with h5py.File(file_path, 'r') as file:
            return len(file.keys()) == 0
    except OSError:
        return True


class data_prep(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, test_dataset,):#,
                 #val_batch_sampler):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        train_sampler = None
        train_loader = DataLoader(
            dataset=self.train_dataset,
            num_workers=configs["training"]["num_workers"],
            shuffle=False,
            batch_size=configs["training"]["batch_size"],
            batch_sampler=train_sampler,
            )
        return train_loader

    def val_dataloader(self):
        val_sampler = None
        eval_loader = DataLoader(
            dataset=self.eval_dataset,
            num_workers=configs["training"]["num_workers"],
            shuffle=False,
            sampler=val_sampler,
            batch_size=configs["training"]["batch_size_val"],
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
    config.model = args.model

    conf_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "confs/default.yaml",
    )

    with open(conf_file_path, "r") as f:
        configs = yaml.safe_load(f)
    
    base_log_dir = os.path.join("/work", "unegi2s", "epochs_" + str(configs["training"]["max_epoch"]))
    pred_save_dir = base_log_dir

    if not os.path.exists(base_log_dir):
        os.mkdir(base_log_dir)

    if configs["student_teacher_model"]:
        base_log_dir = os.path.join(base_log_dir, "student_teacher")
        pred_save_dir = base_log_dir
        if not os.path.exists(base_log_dir):
            os.mkdir(base_log_dir)
        if not os.path.exists(pred_save_dir):
            os.mkdir(pred_save_dir)
    
    if configs["augment_data"]:
        base_log_dir = os.path.join(base_log_dir, "mixup_specaugment")
        if not os.path.exists(base_log_dir):
            os.mkdir(base_log_dir)
        pred_save_dir = base_log_dir
        if not os.path.exists(pred_save_dir):
            os.mkdir(pred_save_dir)

    log_dir = os.path.join(base_log_dir, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = os.path.join(log_dir, config.model)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    pred_save_dir = os.path.join(pred_save_dir, "predictions")
    if not os.path.exists(pred_save_dir):
        os.mkdir(pred_save_dir)

    config.pred_save_dir = pred_save_dir

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

    ################MODEL########################################

    checkpoint_callback = ModelCheckpoint(
        monitor="val/obj_metric",
        dirpath="checkpoints/l-{epoch:d}-{mAP:.3f}",
        # filename='l-{epoch:d}-{mAP:.3f}',
        save_top_k=2,
        mode="max",
    )
    early_stop = EarlyStopping(
                monitor="val/obj_metric",
                patience=configs["training"]["early_stop_patience"],
                verbose=True,
                mode="max",
            )

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name=args.model)
    trainer = pl.Trainer(
        deterministic=False,
        # accelerator="cpu",  # For running locally,
        accelerator="gpu",
        #gpus=None,  # For running locally,
        gpus=[0],
        max_epochs=configs["training"]["max_epoch"],
        # max_epochs=configs["training"]["max_epoch"],
        auto_lr_find=False,
        sync_batchnorm=True,
        num_sanity_val_steps=0,
        # resume_from_checkpoint = config.resume_checkpoint,
        gradient_clip_val=1.0,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop, TQDMProgressBar(refresh_rate=1000)]
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
            "window_size": WIN_LENGTH,
            "hop_size": HOP_LENGTH,
            "mel_bins": N_MELS,
            "fmin": F_MIN,
            "fmax": F_MAX,
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
   
    audio_class_model = AudioClassification(sed_model=sed_model, config=config,
                                            prefix_folder=configs["data"]["prefix_folder"])
   

    
    ################ DATA ########################################
    hdf_file_path = configs["hdf_file"]

    if is_hdf5_empty(hdf_file_path):
        data = prepare_all_data(configs["data"])
        create_hdf_file(data, hdf_file_path, sr=SAMPLE_RATE)

    h5py_file = h5py.File(hdf_file_path, "r")

    train_data = h5py_file["train"]
    eval_data = h5py_file["eval"]
    test_data = h5py_file["test"]

    train_dataset_strong2weak = SEDDataset(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        data=train_data["strong"],
        config=configs,
        device=device,
    )

    train_dataset_synth2weak = SEDDataset(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        data=train_data["synth"],
        config=configs,
        device=device,
    )

   
    train_dataset_weak = SEDDataset(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        data=train_data["weak"],
        config=configs,
        device=device,
    )

    eval_dataset_synth2weak = SEDDataset(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        data=eval_data["synth"],
        config=configs,
        device=device,
    )

    eval_dataset_weak = SEDDataset(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        data=eval_data["weak"],
        config=configs,
        device=device,
    )


    test_dataset2weak = SEDDataset(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        data=test_data["strong"],
        config=configs,
        device=device,
    )


############# PSEUDO LABELLING #########################
    unlabelled_dataset = UnlabelledDataset(dirpath = os.path.join(configs["data"]["prefix_folder"], configs["data"]["unlabeled_folder"]),
                                                                  num_samples=NUM_SAMPLES,
        config=configs)
   
    pl_total_train_weak_data = [train_dataset_strong2weak, train_dataset_synth2weak, train_dataset_weak]
    pl_train_weak = torch.utils.data.ConcatDataset(pl_total_train_weak_data)
    #pl_train_weak_subset = Subset(pl_train_weak, np.arange(5))
    pl_total_val_weak_data = [eval_dataset_synth2weak, eval_dataset_weak]
    pl_val_weak = torch.utils.data.ConcatDataset(pl_total_val_weak_data)
    #for a in pl_val_weak:
    #    print(a["target"].shape)
    #pl_val_weak_subset = Subset(pl_val_weak, np.arange(5))
    pl_test = test_dataset2weak
    #pl_test_subset = Subset(test_dataset2weak, np.arange(5))

    pseudo_labelled_data = data_prep(pl_train_weak, pl_val_weak,
                                     pl_test)
   
    epoch_len = min(
            [
                len(pl_total_train_weak_data[indx])
                // (
                    configs["training"]["batch_size"]
                    * configs["training"]["accumulate_batches"]
                )
                for indx in range(len(pl_total_train_weak_data))
            ]
        )
    opt = torch.optim.Adam(sed_model.parameters(), configs["opt"]["lr"], betas=(0.9, 0.999))
    exp_steps = configs["training"]["n_epochs_warmup"] * epoch_len
    exp_scheduler = {
            "scheduler": ExponentialWarmup(opt, configs["opt"]["lr"], exp_steps),
            "interval": "step",
        }

    sed_teacher = None
    model = SEDWrapper(
        sed_model=sed_model,
        sed_teacher = sed_teacher,
        config=config,
        prefix_folder=configs["data"]["prefix_folder"],
        opt = opt, 
        scheduler = exp_scheduler
    )

    model.learning_rate = LEARNING_RATE

   
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


    trainer.fit(audio_class_model,
                pseudo_labelled_data.train_dataloader(),
                pseudo_labelled_data.val_dataloader())
   
    trainer.test(audio_class_model,
                 pseudo_labelled_data.test_dataloader())

           
    unlabelled_loader = DataLoader(
            dataset=unlabelled_dataset,
            shuffle=False,
            batch_size=2,
        )

    results = []
    threshold = 0.5
    for batch in unlabelled_loader:
        predictions = audio_class_model.predict_step(batch)
        # Collect the batch information along with predictions
        for waveform, filename, prediction in zip(batch["waveform"], batch["audio_name"],
                                                  predictions):
            prediction = prediction.detach().cpu().numpy()
            labels = [1 if x >= threshold else 0 for x in prediction]
            labels2int = [i for i, val in enumerate(labels) if val == 1]
            results.append({
            "audio_name": filename,
            "waveform": waveform,
            "target": labels2int
        })
    unlabelled_h5_file = os.path.join(configs["data"]["prefix_folder"], config.model, configs["unlabelled_hdf_file"])
    file_unlabelled = h5py.File(unlabelled_h5_file, "w")
    for element in results:
        group = file_unlabelled.create_group(os.path.basename(element["audio_name"]))
        group.attrs["folder_path"] = os.path.dirname(element["audio_name"])
        group.attrs["class_labels"] = element["target"]
        group.create_dataset("waveform", data=group.create_dataset("waveform", data=element["waveform"]))
    file_unlabelled.close()
    
    
    print(results[0])

    h5py_file.close()
    end_time = time.time()
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

    logging.info(
        f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
    )

    logging.info(f"Total time: {end_time - start_time}")
