import pytorch_lightning as pl
import os, sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchaudio.transforms import Spectrogram
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision, MultilabelConfusionMatrix
import numpy as np
from utils import get_loss_func
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import yaml
import sed_scores_eval
from torchcontrib.optim import SWA
import torch.optim as optim

class AudioClassification(pl.LightningModule):
    def __init__(self, sed_model, config, prefix_folder):
        super().__init__()
        self.sed_model = sed_model
        self.config = config
        self.frame_loss_func = get_loss_func(config.loss_type_frame)
        self.class_loss_func = get_loss_func(config.loss_type_class)
        self.writer = SummaryWriter()
        self.learning_rate = config.learning_rate
        print(self.device)

        self.multilabel_f1_val = MultilabelF1Score(num_labels=len(self.config.classes2id), average='macro').to(self.device)
        self.multilabel_f1_test = MultilabelF1Score(num_labels=len(self.config.classes2id), average='macro').to(self.device)

        self.mAP_val = MultilabelAveragePrecision(num_labels=len(self.config.classes2id), average="macro", thresholds=None)
        self.mAP_test = MultilabelAveragePrecision(num_labels=len(self.config.classes2id), average="macro", thresholds=None)
       
        self.prefix_folder = prefix_folder
        #self.filename_eval = self.df_eval.filename.unique()
        #self.filename_test = self.df_test.filename.unique()
        self.prediction_dfs = []
        conf_file_path = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "confs/default.yaml",
                            )
        with open(conf_file_path, "r") as f:
            self.configs = yaml.safe_load(f)
        # for weak labels we simply compute f1 score
        self.val_decoded_pred = {}
        self.test_decoded_pred = {}
        self.pred_clip_test = torch.Tensor()
        self.labels_frame2class_test = torch.Tensor()


    def forward(self, x, mix_lambda=None):
        
        if isinstance(x, list):
            tensor_x = torch.stack(x)
            output_dict = self.sed_model(tensor_x, mix_lambda)
        else:
            output_dict = self.sed_model(x)
        return output_dict["clipwise_output"], output_dict["framewise_output"]

    def training_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        batch_num = len(batch)
        audio_name, audio, label = batch["audio_name"], batch["waveform"], batch["target"]
        #mix_lambda = torch.from_numpy(get_mix_lambda(0.5, len(batch["waveform"]))).to(self.device_type)
        mix_lambda = None
        pred_clip, _ = self(audio, mix_lambda=mix_lambda)
        
        target_class = label.float()
        loss = self.class_loss_func(pred_clip, target_class)
        
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        audio_name, audio, labels = batch["audio_name"], batch["waveform"], batch["target"]
        batch_num = len(batch)
        pred_clip, _ = self(audio)
        #print(pred_clip.shape)
        labels_weak = labels.float()
        val_loss = self.class_loss_func(pred_clip, labels_weak)
        self.log('val/loss', val_loss)
        # Calculate F1 Score
        f1_val = self.multilabel_f1_val(pred_clip, labels.long())
        self.log('val/multilabel_f1', f1_val)
        mAP_val = self.mAP_val(pred_clip, labels.long())
        self.log('val/obj_metric', mAP_val)
        
    
    def validation_epoch_end(self, outputs):
        f1_multilabel = self.multilabel_f1_val.compute()
        mAP_val = self.mAP_val.compute()
        self.device_type = next(self.parameters()).device

        ground_truth = sed_scores_eval.io.read_ground_truth_events(os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["synth_val_tsv"]))
        audio_durations = sed_scores_eval.io.read_audio_durations(os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["synth_val_dur"]))
        ground_truth = {
                audio_id: gt for audio_id, gt in ground_truth.items()
                if len(gt) > 0
            }
        audio_durations = {
                audio_id: audio_durations[audio_id]
                for audio_id in ground_truth.keys()
            }
        self.log('val/multilabel_f1', f1_multilabel)
        self.log('val/obj_metric', mAP_val)
        self.multilabel_f1_val.reset()
        self.mAP_val.reset()
    
    def test_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        audio_name, audio, labels = batch["audio_name"], batch["waveform"], batch["target"]
        batch_num = len(batch)
        pred_clip, _ = self(audio)

        self.multilabel_f1_test(pred_clip, labels.long())
        self.mAP_test(pred_clip, labels.long())

    def test_epoch_end(self, outputs):
        ground_truth = sed_scores_eval.io.read_ground_truth_events(os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["test_tsv"]))
        audio_durations = sed_scores_eval.io.read_audio_durations(os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["test_dur"]))
        ground_truth = {
                audio_id: gt for audio_id, gt in ground_truth.items()
                if len(gt) > 0
            }
        audio_durations = {
                audio_id: audio_durations[audio_id]
                for audio_id in ground_truth.keys()
            }
        multilabel_f1_test = self.multilabel_f1_test.compute()
        mAP_test = self.mAP_test.compute()
        self.log("test/multilabel f1", multilabel_f1_test, on_epoch=True, prog_bar=True)
        self.log("test/mAP test", mAP_test, on_epoch=True, prog_bar = True)
        self.multilabel_f1_test.reset()
        self.mAP_test.reset()

    def predict_step(self, batch):
        clip_pred, _ = self(batch["waveform"])
        return clip_pred

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.05,
        )
