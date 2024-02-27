# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# The Model Training Wrapper
import numpy as np
import librosa
import os
import sys
import math
import bisect
import pickle
import config
from numpy.lib.function_base import average
from pathlib import Path
from sklearn import metrics
import soundfile as sf
from inference import batched_decode_preds, log_sedeval_metrics
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from desed_task.evaluation.evaluation_measures import compute_sed_eval_metrics

# import tfplot
import matplotlib.pyplot as plt

# import seaborn as sns
import pandas as pd
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision, MultilabelConfusionMatrix
import config

from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
    compute_psds_from_scores
)


from utils import get_loss_func, get_mix_lambda, d_prime
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch.distributed as dist
from torchlibrosa.stft import STFT, ISTFT, magphase
import pytorch_lightning as pl
from utils import do_mixup, get_mix_lambda, do_mixup_label
import random
from torchmetrics import Precision, Recall
from torch import tensor
from torchmetrics.functional import f1_score
import torchmetrics
from tqdm import tqdm
import yaml
import sed_scores_eval
from torchcontrib.optim import SWA

def frame_to_time(frame):
        hop_length = 320
        sr = 16000
        return (frame * hop_length) / sr


class SEDWrapper(pl.LightningModule):
    def __init__(self, sed_model, config, prefix_folder):
        super().__init__()
        self.sed_model = sed_model
        self.config = config
        # self.dataset = dataset
        self.frame_loss_func = get_loss_func(config.loss_type_frame)
        self.class_loss_func = get_loss_func(config.loss_type_class)
        self.writer = SummaryWriter()
        self.learning_rate = config.learning_rate
        self.get_weak_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
            len(self.config.classes2id), average="macro"
        ).to(self.device)
        self.multilabel_f1_val = MultilabelF1Score(num_labels=len(self.config.classes2id), average='macro').to(self.device)
        self.multilabel_f1_test = MultilabelF1Score(num_labels=len(self.config.classes2id), average='macro').to(self.device)
        self.multilabel_f1_test_classes = MultilabelF1Score(num_labels=len(self.config.classes2id), average='none').to(self.device)
        self.mAP_val = MultilabelAveragePrecision(num_labels=len(self.config.classes2id), average="macro", thresholds=None).to(self.device)
        self.mAP_test = MultilabelAveragePrecision(num_labels=len(self.config.classes2id), average="macro", thresholds=None).to(self.device)
        self.mAP_test_classes = MultilabelAveragePrecision(num_labels=len(self.config.classes2id), average="none", thresholds=None).to(self.device)
        self.confmat = MultilabelConfusionMatrix(num_labels=len(self.config.classes2id), threshold = 0.1).to(self.device)
        #self.df_eval = df_eval
        #self.df_test = df_test
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
        self.pred_clip_test = torch.Tensor().to(self.device)
        self.labels_frame2class_test = torch.Tensor().to(self.device)
        self.get_weak_f1_seg_macro_test = torchmetrics.classification.f_beta.MultilabelF1Score(
                            len(self.config.classes2id), average="macro"
                                    ).to(self.device)

        self.thresholds = self.configs["training"]["thresholds"]
        test_n_thresholds = self.configs["training"]["n_test_thresholds"]
        self.test_thresholds = np.arange(1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
                                        )
        #print(self.test_thresholds)
        self.test_scores_postprocessed = {}
        self.test_psds_buffer = pd.DataFrame()
        self.test_decoded_buffer = pd.DataFrame()
        self.val_scores_postprocessed = {}
        self.val_psds_buffer = pd.DataFrame()
        self.val_decoded_buffer = pd.DataFrame()
        self.val_buffer_synth = {}
        self.test_buffer_synth = {}
        self.test_psds_buffer = {k: pd.DataFrame() for k in self.test_thresholds}
        self.decoded_05_buffer = pd.DataFrame()
        self.test_scores_raw_buffer = {}
        #self.test_scores_postprocessed_buffer = {}
        """for th in self.test_thresholds:
            self.test_buffer_synth[th] = pd.DataFrame()
            self.val_buffer_synth[th] = pd.DataFrame()
            self.val_decoded_pred[th] = pd.DataFrame()
            self.test_decoded_pred[th] = pd.DataFrame()"""

        #print(self.test_decoded_pred.keys())
    

    def evaluate_metric(self, pred_target_dict):
        ap = []
        pred = pred_target_dict["pred"]
        target = pred_target_dict["target"]

        pred_class = pred[0].cpu()
        pred_frame = pred[1]
        target_class = target[1].cpu()
        #target = target[0]
        #target_frame = target[1]

        if self.config.dataset_type == "audioset":
            """flat_ans = ans.flatten()
            flat_pred = pred.flatten()
            flat_pred_class = pred_class.flatten()"""
            mAP_new = np.mean(average_precision_score(target_class, pred_class, average = None))
            #mAUC_new = np.mean(roc_auc_score(target_class, pred_class, average = None))
            # ans = ans.astype(np.float32)
            pred_class = pred_class
            # pred_frame = pred_frame.astype(np.float32)
            #target = target_class
            # target_frame = target_frame.astype(np.float32)

            #reshaped_pred_frame = pred_frame.transpose(0, 2, 1)
            #reshaped_target = target.transpose(0, 2, 1)
            # reshaped_pred_class = pred_class.transpose(1,0)
            # reshaped_target_class = target_class.transpose(1,0)

            threshold = 0.5
            predicted_labels = pred_class > threshold
            #reshaped_target_class = target
            ap_values = []
            auc_scores = []
            f1_scores = []
            """for class_idx in range(10):
                class_target_frame = reshaped_target[:, class_idx, :]
                class_pred_frame = reshaped_pred_frame[:, class_idx, :]
                ap = average_precision_score(
                    class_target_frame, class_pred_frame, average=None
                )
                ap_values.append(ap)
                # auc = roc_auc_score(class_ans, class_pred)
                # auc_scores.append(auc)

                # f1 = f1_score(reshaped_target_class[class_idx, :], predicted_labels[class_idx, :], task = "multiclass")
                # f1_scores.append(f1)"""

            #mAP = np.mean(ap_values)
            # mAUC = np.mean(auc_scores)
            #f1 = self.f1_score(predicted_labels, reshaped_target_class).to(self.device)

            # mAP = np.mean(average_precision_score(flat_ans, flat_pred, average = None))
            # mAUC = np.mean(roc_auc_score(flat_ans, flat_pred, average = None))
            # dprime = d_prime(mAUC)
            mAUC = 0
            dprime = 0
            # precision = precision_score(ans, pred, average='macro')
            # recall = recall_score(ans, pred, average='macro')
            # f1 = f1_score(flat_ans, flat_pred, average='macro', task= "multiclass")
            # f1 = f1_score(torch.tensor(pred), torch.tensor(ans), num_classes=config.classes_num, task= "multiclass")
            # pred_int = pred.astype(int)
            # ans_int = ans.astype(int)
            # pred_tensor = torch.from_numpy(pred)
            # pred_categorical = F.one_hot(pred_tensor, num_classes=ans.shape[1])
            # pred_tensor = torch.from_numpy(pred_categorical)
            # precision = precision_score(ans_int, pred_int, average='macro')
            # recall = recall_score(ans_int, pred_int, average='macro')
            # f1 = f1_score(ans_int, pred_int, average='macro')
            # if len(ans.shape) > 1:  # Check if ans is in "multilabel-indicator" format
            #    ans_cm = np.argmax(ans, axis=1)
            # if len(pred.shape) > 1:  # Check if pred is in "multilabel-indicator" format
            #    pred_cm = np.argmax(pred, axis=1)
            # cm = MulticlassConfusionMatrix(num_classes=10)
            # confusion_matrix = cm(torch.tensor(pred), torch.tensor(ans))
            return {
                "mAP": mAP_new,
                "mAUC": mAUC,
                "dprime": dprime,
                # "precision": precision, "recall": recall,
                #"f1": f1,
            }  # , "confusion_matrix": confusion_matrix}
        else:
            acc = accuracy_score(ans, np.argmax(pred, 1))
            return {"acc": acc}

    def forward(self, x, mix_lambda=None):
        if isinstance(x, list):
            tensor_x = torch.stack(x)
            output_dict = self.sed_model(tensor_x, mix_lambda)
        else:
            output_dict = self.sed_model(x, mix_lambda)
        return output_dict["clipwise_output"], output_dict["framewise_output"]

    def inference(self, x):
        self.device_type = next(self.parameters()).device
        self.eval()
        x = torch.from_numpy(x).float().to(self.device_type)
        output_dict = self.sed_model(x, None, True)
        #for key in output_dict.keys():
        #    output_dict[key] = output_dict[key].detach().cpu().numpy()
        return output_dict

    def training_step(self, batch, batch_idx):
        indx_synth, indx_weak = self.configs["training"]["batch_sizes"]
        self.device_type = next(self.parameters()).device
        batch_num = len(batch)
        audio_name, audio, label = batch["audio_name"], batch["waveform"], batch["target"]
        strong_mask = torch.zeros(batch_num).bool()
        weak_mask = torch.zeros(batch_num).bool()
        strong_mask[:indx_synth] = 1
        weak_mask[indx_synth : indx_weak + indx_synth] = 1
        ##if self.config.dataset_type == "audioset":
        ##mix_lambda = torch.from_numpy(get_mix_lambda(0.5, len(batch["waveform"]))).to(self.device_type)
        # else:
        mix_lambda = None

        # Another Choice: also mixup the target, but AudioSet is not a perfect data
        # so "adding noise" might be better than purly "mix"
        # batch["target_frames"] = do_mixup_label(batch["target_frames"])
        # batch["target_frames"] = do_mixup(batch["target_frames"], mix_lambda)
        pred_clip, pred_frame = self(audio, mix_lambda)
        # loss = self.loss_func(pred_clip, batch["target_frames"])
        # pred_frame = pred_frame.float()

        

        #target = batch["target"].float()
        #target_classes = batch["target_classes"]
        target_frame = torch.stack(label[:indx_synth]).float()
        target_class = torch.stack(label[indx_synth : indx_weak + indx_synth]).float()
        loss_frame = self.frame_loss_func(pred_frame[:indx_synth], target_frame)
        loss_class = self.class_loss_func(pred_clip[indx_synth : indx_weak + indx_synth], target_class)
        #loss_frame = self.frame_loss_func(pred_frame[strong_mask], target[strong_mask])
        #loss_class = self.class_loss_func(pred_clip[weak_mask], target[weak_mask])
        loss = loss_frame + loss_class
        self.log("train/loss_strong", loss_frame, on_epoch=True, prog_bar=True)
        self.log("train/loss_weak", loss_class, on_epoch=True, prog_bar=True)
        self.log("train/train_loss", loss, on_epoch=True, prog_bar=True)
        self.writer.add_scalar('train_loss',
                                        loss)
        return loss

    # def training_epoch_end(self, outputs):
    # Change: SWA, deprecated
    # for opt in self.trainer.optimizers:
    #     if not type(opt) is SWA:
    #         continue
    #     opt.swap_swa_sgd()
    # self.dataset.generate_queue()

    def validation_step(self, batch, batch_idx):
        audio_name, audio, labels = batch["audio_name"], batch["waveform"], batch["target"]
        batch_num = len(batch)
        pred_clip, pred_frame = self(audio)
        target = labels
        
        mask_weak = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["weak_folder"])))
                    for x in batch["audio_name"]
                ]
            )
            .to(batch["waveform"])
            .bool()
        )

        if torch.any(mask_weak):
            labels_weak = (torch.sum(labels[mask_weak], -1) >= 1).float()
            #print(pred_clip[mask_weak].dtype)
            #print(labels[mask_weak].dtype)
            loss_class = self.class_loss_func(pred_clip[mask_weak], labels[mask_weak].float())
            self.log("val/weak/loss_weak", loss_class)

            # accumulate f1 score for weak labels
            #print("Pred clip")
            #print(pred_clip[mask_weak].shape)
            #print("labels")
            #print(labels[mask_weak].float().shape)
            self.get_weak_f1_seg_macro(
                pred_clip[mask_weak], labels[mask_weak].float())
            self.multilabel_f1_val(pred_clip[mask_weak], labels[mask_weak].long())
            self.mAP_val(pred_clip[mask_weak], labels[mask_weak].long())

        mask_synth = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["synth_val_folder"])))
                    for x in  batch["audio_name"]
                ]
            )
            .to(batch["waveform"])
            .bool()
        )

        if torch.any(mask_synth):
            if config.loss_type_frame == "clip_bce":
                loss_strong = self.frame_loss_func(
                    pred_frame[mask_synth].float(), labels[mask_synth].float()
                                                        )
            else:
                loss_strong = self.frame_loss_func(
                pred_frame[mask_synth], labels[mask_synth]
            )

            self.log("val/synth/loss_strong", loss_strong)
            labels_frame2class = torch.any(labels == 1, dim=1).int().squeeze()
            
            if len(labels_frame2class.shape) < 2:
                labels_frame2class = labels_frame2class.unsqueeze(0) 
            
            self.mAP_val(pred_clip, labels_frame2class)
            #print(mask_weak)
            #print(pred_clip[mask_weak])
            # accumulate f1 score for weak labels

            filenames_synth = [
                x
                for x in audio_name
                if Path(x).parent == Path(os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["synth_val_folder"]))
            ]
            filenames_synth = [os.path.basename(file_path) for file_path in filenames_synth]
            
            (
            scores_raw_strong, scores_postprocessed_strong,
            decoded_strong,
            ) = batched_decode_preds(pred_frame,filenames_synth, hop_length = self.configs["feats"]["hop_length"],
                    sr = self.configs["feats"]["sample_rate"], median_filter=self.configs["training"]["median_window"],
                    thresholds=self.test_thresholds)
            self.val_scores_postprocessed.update(scores_postprocessed_strong)

            for th in self.test_thresholds:
                self.val_buffer_synth[th] = pd.concat([self.val_buffer_synth[th], decoded_strong[th]], ignore_index=True)
                self.val_decoded_pred[th] = pd.concat([self.val_decoded_pred[th], decoded_strong[th]], 
                                                                                                  ignore_index = True)

            """decoded_strong = batched_decode_preds(
            pred_frame, filenames_synth, thresholds=0.4, median_filter=7, pad_indx=None,
            hop_length = self.configs["feats"]["hop_length"], sr = self.configs["feats"]["sample_rate"]
            )
            self.val_decoded_pred = pd.concat([self.val_decoded_pred, decoded_strong], 
                                          ignore_index = True)"""


        """for filename, waveform in zip(batch["audio_name"], batch["waveform"]):
            prediction_df = self.prediction_for_clip(os.path.basename(filename),
                                                clip=waveform,
                                                threshold=0.5)
            self.prediction_dfs.append(prediction_df)"""
        # self.log()
        # pred_target_dict = {"pred" : [pred_clip.detach(), pred_frame.detach()], "target": [batch["target_classes"].detach(), batch["target_frames"].detach()] }
        return [
            pred_clip.detach(),
            pred_frame.detach(),
            batch["target"].detach(),
            #batch["target_frames"].detach(),
        ]

    def validation_epoch_end(self, validation_step_outputs):
        weak_f1_macro = self.get_weak_f1_seg_macro.compute()
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
                            audio_id: audio_durations[audio_id] for audio_id in ground_truth.keys()
                        }
        psds1_sed_scores_eval = compute_psds_from_scores(
                            self.val_scores_postprocessed,
                            ground_truth,
                            audio_durations,
                            dtc_threshold=0.7,
                            gtc_threshold=0.7,
                            cttc_threshold=None,
                            alpha_ct=0,
                            alpha_st=1,
                            save_dir=os.path.join("psds", "scenario1"),)
        intersection_f1_macro = compute_per_intersection_macro_f1(
                            self.val_buffer_synth,
                            os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["synth_val_tsv"]),
                            os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["synth_val_dur"]),
                                                            )

        self.log("psds", psds1_sed_scores_eval, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("intersection_f1_macro", intersection_f1_macro, on_epoch=True, prog_bar=True, sync_dist=False)
        
        # print(target.shape)
        pred_classes = torch.cat([d[0] for d in validation_step_outputs], dim=0)
        pred_frame = torch.cat([d[1] for d in validation_step_outputs], dim=0)
        target = torch.tensor([]).to(self.device_type) 
        for d in validation_step_outputs:
            if len(d[2].shape) == 3:
                target_class = torch.any(d[2] == 1, dim=1).int().to(self.device_type)
            elif len(d[2].shape) == 2:
                target_class = d[2]
            target = torch.cat((target, target_class), dim=0).to(self.device_type)
        if torch.cuda.device_count() > 1:
            gather_pred_classes = [
                torch.zeros_like(pred_classes) for _ in range(dist.get_world_size())
            ]
            gather_pred_frame = [
                torch.zeros_like(pred_frame) for _ in range(dist.get_world_size())
            ]
            #gather_target_classes = [
            #    torch.zeros_like(target_classes) for _ in range(dist.get_world_size())
            #]
            gather_target_frame = [
                torch.zeros_like(target) for _ in range(dist.get_world_size())
            ]
            dist.barrier()

        if self.config.dataset_type == "audioset":
            class_wise_f1 = {}
            metric_dict = {
                "mAP": 0.0,
                "class_wise_f1": class_wise_f1,
                "mAUC": 0.0,
                "dprime": 0.0,
                # "precision": 0.,
                # "recall": 0.,
                "f1": 0.0,
            }
            sedeval_metrics = log_sedeval_metrics(
            self.val_decoded_pred, os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["synth_val_tsv"],
        ))
            synth_event_f1_macro = sedeval_metrics[0]
            synth_event_f1 = sedeval_metrics[1]

            if isinstance(sedeval_metrics, dict):
                for keys, values in sedeval_metrics[2].items():
                    if keys == "class_wise":
                        for class_name, metrics in values.items():
                            class_wise_f1[class_name] = values[class_name]["f_measure"]["f_measure"]
            
            
        else:
            metric_dict = {"acc": 0.0}
        """if torch.cuda.device_count() > 1:
            dist.all_gather(gather_pred, pred)
            dist.all_gather(gather_target, target)
            if dist.get_rank() == 0:
                gather_pred = torch.cat(gather_pred, dim = 0).cpu().numpy()
                gather_target = torch.cat(gather_target, dim = 0).cpu().numpy()
                if self.config.dataset_type == "scv2":
                    gather_target = np.argmax(gather_target, 1)
                metric_dict = self.evaluate_metric(gather_pred, gather_target)
                print(self.device_type, dist.get_world_size(), metric_dict, flush = True)
        
            if self.config.dataset_type == "audioset":
                self.log("mAP", metric_dict["mAP"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
                self.log("mAUC", metric_dict["mAUC"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
                self.log("dprime", metric_dict["dprime"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            else:
                self.log("acc", metric_dict["acc"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            dist.barrier()
        else:"""
        gather_pred_classes = pred_classes
        gather_pred_frame = pred_frame.cpu().numpy()
        #gather_target_classes = target_classes
        gather_target = target.cpu().numpy()
        if self.config.dataset_type == "scv2":
            gather_target = np.argmax(gather_target, 1)
            metric_dict = self.evaluate_metric(
                gather_pred_classes, gather_target_classes
            )
            print(self.device_type, metric_dict, flush=True)

        pred_target_dict = {
            "pred": [gather_pred_classes, gather_pred_frame],
            "target": [gather_target, target],
        }

        if self.config.dataset_type == "audioset":
            metric_dict = self.evaluate_metric(pred_target_dict)
            metric_dict["class_wise_f1"] = class_wise_f1
            self.log(
                "mAP", metric_dict["mAP"], on_epoch=True, prog_bar=True, sync_dist=False
            )
            self.log(
                "weak_f1", weak_f1_macro, on_epoch=True, prog_bar=True, sync_dist=False
            )
            self.log("f1_multilabel", f1_multilabel, on_epoch = True, prog_bar = True)
            self.log("mAP val", mAP_val, on_epoch = True, prog_bar = True)
            #self.log("class_wise_f1", metric_dict["class_wise_f1"], on_epoch=True, prog_bar=True)
            self.log("event_f1_macro", synth_event_f1_macro, on_epoch=True, prog_bar=False)
            self.log("event_f1", synth_event_f1, on_epoch=True, prog_bar=True)
            """self.log(
                "mAUC",
                metric_dict["mAUC"],
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
            )"""
            """self.log(
                "dprime",
                metric_dict["dprime"],
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
            )"""
            # self.log("precision", metric_dict["precision"], on_epoch = True, prog_bar=True, sync_dist=False)
            # self.log("recall", metric_dict["recall"], on_epoch = True, prog_bar=True, sync_dist=False)
            #
            #self.log(
            #    "f1", metric_dict["f1"], on_epoch=True, prog_bar=True, sync_dist=False
            #)
            #self.df_eval["event_label"] = self.df_eval["event_label"].map(config.id2classes)
            #self.prediction_dfs = self.prediction_dfs[0]
            #print(self.prediction_dfs)
            #print("********************")
            #print(self.df_eval)
            ##event_metrics_s, segment_metrics_s = compute_sed_eval_metrics(self.prediction_dfs, self.df_eval)
            ##class_wise_f_measure = event_metrics_s.results()["class_wise_average"]["f_measure"]["f_measure"] * 100
            ##class_wise_f_measure = round(class_wise_f_measure, 2)
            #print(class_wise_f_measure)
            ##overall_wise_f_measure = event_metrics_s.results()["overall"]["f_measure"]["f_measure"] * 100
            ##overall_wise_f_measure = round(overall_wise_f_measure, 2)
            #print(overall_wise_f_measure)
        else:
            self.log(
                "acc", metric_dict["acc"], on_epoch=True, prog_bar=True, sync_dist=False
            )
        self.get_weak_f1_seg_macro.reset()
        self.multilabel_f1_val.reset()
        self.mAP_val.reset()
        metric_dict = {"mAP": 0.0}

    def time_shifting(self, x, shift_len):
        shift_len = int(shift_len)
        new_sample = torch.cat([x[:, shift_len:], x[:, :shift_len]], axis=1)
        return new_sample

    def test_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        preds = []
        audio_name, audio, labels = batch["audio_name"], batch["waveform"], batch["target"]
        batch_num = len(batch)
        pred_clip, pred_frame = self(audio)
        pred_clip = pred_clip.to(self.device)
        #print(pred_clip)
        labels_frame2class = torch.any(labels == 1, dim=1).int().squeeze().to(self.device)
        if len(labels_frame2class.shape) < 2:
            labels_frame2class = labels_frame2class.unsqueeze(0).to(self.device)
        self.labels_frame2class_test = torch.cat((self.labels_frame2class_test.to(self.device), labels_frame2class.to(self.device)), dim=0)
        self.get_weak_f1_seg_macro_test(pred_clip.to(self.device), labels_frame2class.to(self.device))
        self.multilabel_f1_test(pred_clip, labels_frame2class)
        self.multilabel_f1_test_classes(pred_clip, labels_frame2class)
        self.mAP_test(pred_clip, labels_frame2class)
        self.mAP_test_classes(pred_clip, labels_frame2class)
        if len(pred_clip.shape) < 2:
            pred_clip = pred_clip.unsqueeze(0)

        self.pred_clip_test = torch.cat((self.pred_clip_test.to(self.device), pred_clip.to(self.device)), dim=0)
        # time shifting optimization
        """if self.config.fl_local or self.config.dataset_type != "audioset":
            shift_num = 1  # framewise localization cannot allow the time shifting
        else:
            shift_num = 10
        for i in range(shift_num):
            pred_clip, pred_frame = self(batch["waveform"])
            # preds.append(pred_clip.unsqueeze(0))
            preds.append(pred_frame)
            batch["waveform"] = self.time_shifting(
                batch["waveform"], shift_len=100 * (i + 1)
            )"""
        """for filename, waveform in zip(batch["audio_name"], batch["waveform"]):
            prediction_df = self.prediction_for_clip(os.path.basename(filename),
                                                clip=waveform,
                                                threshold=0.5)
            self.prediction_dfs.append(prediction_df)"""
        
        filenames_test = [
                x
                for x in batch["audio_name"]
                if Path(x).parent == Path(os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["val_folder"]))
            ]
        filenames_test = [os.path.basename(file_path) for file_path in filenames_test]
        
        """decoded_strong = batched_decode_preds(
        pred_frame, filenames_test, thresholds=0.4, median_filter=7, pad_indx=None,
        hop_length = self.configs["feats"]["hop_length"], sr = self.configs["feats"]["sample_rate"]
        )
        
        self.test_decoded_pred = pd.concat([self.test_decoded_pred, decoded_strong], 
                                          ignore_index = True)"""

        (
         scores_raw_strong, scores_postprocessed_strong,
         decoded_strong,
         ) = batched_decode_preds(pred_frame,
                                  filenames_test,
                                  hop_length = self.configs["feats"]["hop_length"],
                                  sr = self.configs["feats"]["sample_rate"],
                                  median_filter=self.configs["training"]["median_window"],  
                                  thresholds=list(self.test_psds_buffer.keys()) + [.5],
                                  )
        self.test_scores_raw_buffer.update(scores_raw_strong)
        self.test_scores_postprocessed.update(scores_postprocessed_strong)
        
        #print(type(decoded_strong))
        #print(decoded_strong.keys())
        #print(self.test_psds_buffer.keys())
        #print(type(decoded_strong[0.5]))
        for th in self.test_psds_buffer.keys():
            self.test_psds_buffer[th] = pd.concat([self.test_psds_buffer[th], decoded_strong[th]], ignore_index=True)
            #self.test_decoded_pred[th] = pd.concat([self.test_decoded_pred[th], decoded_strong[th]], 
            #                                                                                   ignore_index = True)
        self.decoded_05_buffer = pd.concat([self.decoded_05_buffer, decoded_strong[0.5]])
        
        

        # preds = torch.cat(preds, dim=0)
        # pred = preds.mean(dim = 0)
        """if self.config.fl_local:
            return [
                pred_clip.detach().cpu().numpy(), 
                pred_frame.detach().cpu().numpy(),
                batch["audio_name"],
                #batch["real_len"].cpu().numpy()
            ]
        else:"""
        # pred_target_dict = {"pred" : [pred_clip.detach(), pred_frame.detach()], "target": [batch["target_classes"].detach(), batch["target_frames"].detach()] }
        # pred_target_dict = {"pred" : [pred_clip.detach(), pred_frame.detach()], "target": [batch.labels_class_arr.detach(), batch.labels_frames_arr.detach()] }
        return [
            pred_clip.detach().detach(),
            pred_frame.detach(),
            batch["target"].detach(),
            #batch["target_frames"].detach(),
        ]

    def test_epoch_end(self, test_step_outputs):
        # print(test_step_outputs.shape)
        save_dir = os.path.join("/home/unegi2s/Documents/predictions/", config.model)
        save_dir_raw = os.path.join(save_dir, "scores_raw")
        sed_scores_eval.io.write_sed_scores(self.test_scores_raw_buffer, save_dir_raw)
        print(f"\nRaw scores saved in: {save_dir_raw}")
        save_dir_postprocessed = os.path.join(save_dir, "postprocessed")
        sed_scores_eval.io.write_sed_scores(self.test_scores_postprocessed, save_dir_postprocessed)
        print(f"\nPostprocessed scores saved in: {save_dir_postprocessed}")
        #print(self.test_psds_buffer.keys())
        #print("................")
        #print(self.test_decoded_pred.keys())
        for th in self.test_psds_buffer.keys():
            self.test_psds_buffer[th].to_csv(save_dir + "/predictions_" + str(th) + ".tsv", sep = "\t")
        self.device_type = next(self.parameters()).device
        # pred = torch.cat([d[0] for d in test_step_outputs], dim = 0)
        # target = torch.cat([d[1] for d in test_step_outputs], dim = 0)
        print(".....................................")
        #print(self.pred_clip_test.long().shape)
        #print(self.pred_clip_test.long())
        """conf_mat = self.confmat(self.pred_clip_test.long(), self.labels_frame2class_test.long())
        confusion_matrices = conf_mat.cpu().numpy()
        
        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))

        for i, ax in enumerate(axs.flat):
            conf_matrix = confusion_matrices[i]
            im = ax.imshow(conf_matrix, cmap='Blues')
            for j in range(conf_matrix.shape[0]):
                for k in range(conf_matrix.shape[1]):
                    ax.text(k, j, conf_matrix[j, k], ha='center', va='center', color='black')
            ax.set_title(f'Class {str(config.id2classes[i])}')
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.xaxis.set_ticklabels(['Negative', 'Positive'])
            ax.yaxis.set_ticklabels(['Negative', 'Positive'])
        plt.tight_layout()
        plt.savefig(f'conf_matrix_plots/confusion_matrix_{str(self.configs["training"]["ckpt_epoch"])}_{str(config.model)}.png')
        # Clear the plot for the next iteration
        plt.clf()
        print(conf_mat)"""

        ground_truth = sed_scores_eval.io.read_ground_truth_events(os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["test_tsv"]))
        audio_durations = sed_scores_eval.io.read_audio_durations(os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["test_dur"]))

        ground_truth = {audio_id: gt for audio_id, gt in ground_truth.items()
                        if len(gt) > 0
                         }
        audio_durations = {audio_id: audio_durations[audio_id]
                           for audio_id in ground_truth.keys()
                           }
        #print(self.test_psds_buffer) 
        psds1_psds_eval = compute_psds_from_operating_points(
                                  self.test_psds_buffer,
                                  os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["test_tsv"]),
                                  os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["test_dur"]),
                                  dtc_threshold=0.7,
                                  gtc_threshold=0.7,
                                  alpha_ct=0,
                                  alpha_st=1,
                                  save_dir=os.path.join(save_dir, "scenario1", "psds_eval"),
                                                                                                                                                            )
        #print(self.test_scores_postprocessed)
        psds1_sed_scores_eval = compute_psds_from_scores(
                                self.test_scores_postprocessed,
                                ground_truth,
                                audio_durations,
                                dtc_threshold=0.7,
                                gtc_threshold=0.7,
                                cttc_threshold=None,
                                alpha_ct=0,
                                alpha_st=1,
                                save_dir=os.path.join(save_dir, "scenario1", "sed_eval"),
                                                                
                                )

        psds2_psds_eval = compute_psds_from_operating_points(
                                  self.test_psds_buffer,
                                  os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["test_tsv"]),
                                  os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["test_dur"]),
                                  dtc_threshold=0.1,
                                  gtc_threshold=0.1,
                                  cttc_threshold=0.3,
                                  alpha_ct=0.5,
                                  alpha_st=1,
                                  save_dir=os.path.join(save_dir, "scenario2", "psds_eval"),
                                                 )
        psds2_sed_scores_eval = compute_psds_from_scores(
                                        self.test_scores_postprocessed,
                                        ground_truth,
                                        audio_durations,
                                        dtc_threshold=0.1,
                                        gtc_threshold=0.1,
                                        cttc_threshold=0.3,
                                        alpha_ct=0.5,
                                        alpha_st=1,
                                        save_dir=os.path.join(save_dir, "scenario2", "sed_eval"),
                                                                                                                                                                            )

        event_macro = log_sedeval_metrics(
                              self.decoded_05_buffer,
                              os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["test_tsv"]),
                              save_dir)[0]
        intersection_f1_macro = compute_per_intersection_macro_f1(
                                        {"0.5": self.decoded_05_buffer},
                                        os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["test_tsv"]),
                                        os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["test_dur"]),)
        best_test_result = torch.tensor(max(psds1_psds_eval, psds2_psds_eval))
        

        self.log("psds1", psds1_psds_eval, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("psds1_sed_scores", psds1_sed_scores_eval, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("psds2", psds2_psds_eval, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("psds2_sed_scores", psds2_sed_scores_eval, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("best_psds", best_test_result, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("event_macro", event_macro, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("intersection_f1_macro", intersection_f1_macro, on_epoch=True, prog_bar=True, sync_dist=False)
        pred_classes = torch.cat([d[0] for d in test_step_outputs], dim=0)
        pred_frame = torch.cat([d[1] for d in test_step_outputs], dim=0)
        target = torch.cat([d[2] for d in test_step_outputs], dim=0)
        target_class = torch.any(target == 1, dim=1).int()
        multilabel_f1_test = self.multilabel_f1_test.compute()
        multilabel_f1_test_classes = self.multilabel_f1_test_classes.compute()
        mAP_test = self.mAP_test.compute()
        mAP_test_classes = self.mAP_test_classes.compute()
        #target_frame = torch.cat([d[3] for d in test_step_outputs], dim=0)

        # print("=====================================")
        # print(pred.shape)
        # print(target.shape)
        #self.df_test["event_label"] = self.df_test["event_label"].map(config.id2classes)
        #self.prediction_dfs = self.prediction_dfs[0]
        #self.df_test["event_label"] = self.df_test["event_label"].map(config.id2classes)
        #event_metrics_s, segment_metrics_s = compute_sed_eval_metrics(self.prediction_dfs, self.df_test)
        ##class_wise_f_measure = event_metrics_s.results()["class_wise_average"]["f_measure"]["f_measure"] * 100
        ##class_wise_f_measure = round(class_wise_f_measure, 2)
        #print(class_wise_f_measure)
        ##overall_wise_f_measure = event_metrics_s.results()["overall"]["f_measure"]["f_measure"] * 100
        ##overall_wise_f_measure = round(overall_wise_f_measure, 2)
        """self.log("class-wise-f_measure", class_wise_f_measure, 
                on_epoch=True,
                prog_bar=True,
                sync_dist=False)
        self.log("overall-wise-f_measure", overall_wise_f_measure, 
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,)
        #print("Event-based metric class-wise average metrics (macro average) {:.2f}%".format(event_metrics_s.results()["class_wise_average"]["f_measure"]["f_measure"] * 100))
        #print("Event-based metric overall metrics (micro average) {:.2f}%".format(event_metrics_s.results()["overall"]["f_measure"]["f_measure"] * 100))"""

        """if self.config.fl_local:
            pred = np.concatenate([d[0] for d in test_step_outputs], axis = 0)
            pred_map = np.concatenate([d[1] for d in test_step_outputs], axis = 0)
            audio_name = np.concatenate([d[2] for d in test_step_outputs], axis = 0)
            real_len = np.concatenate([d[3] for d in test_step_outputs], axis = 0)
            heatmap_file = os.path.join(self.config.heatmap_dir, self.config.test_file + "_" + str(self.device_type) + ".npy")
            save_npy = [
                {
                    "audio_name": audio_name[i],
                    "heatmap": pred_map[i],
                    "pred": pred[i],
                    "real_len":real_len[i]
                }
                for i in range(len(pred))
            ]
            np.save(heatmap_file, save_npy)
        else:
            self.device_type = next(self.parameters()).device
            pred = torch.cat([d[0] for d in test_step_outputs], dim = 0)
            target = torch.cat([d[1] for d in test_step_outputs], dim = 0)"""

        # gather_pred = [torch.zeros_like(pred)]
        # gather_target = [torch.zeros_like(target)]
        # dist.barrier()
        if self.config.dataset_type == "audioset":
            class_wise_f1 = {}
            metric_dict = {
                "mAP": 0.0,
                "class_wise_f1": class_wise_f1,
                "mAUC": 0.0,
                "dprime": 0.0,
                # "precision": 0.,
                # "recall": 0.,
                "f1": 0.0,
                # "confusion_matrix" :0.,
            }
        else:
            metric_dict = {"acc": 0.0}
        # dist.all_gather(gather_pred, pred)
        # dist.all_gather(gather_target, target)
        # if dist.get_rank() == 0:
        # gather_pred = torch.cat(gather_pred, dim = 0).cpu().numpy()
        # gather_target = torch.cat(gather_target, dim = 0).cpu().numpy()
        # if self.config.dataset_type == "scv2":
        # gather_target = np.argmax(gather_target, 1)
        # metric_dict = self.evaluate_metric(gather_pred, gather_target)
        # print(self.device_type, dist.get_world_size(), metric_dict, flush = True)
        if self.config.dataset_type == "audioset":
            # self.log("mAP", metric_dict["mAP"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            gather_pred_classes = pred_classes
            gather_pred_frame = pred_frame.cpu().numpy()
            gather_target = target.cpu().numpy()
            #gather_target_classes = target_classes
            #gather_target_frame = target_frame.cpu().numpy()
            pred_target_dict = {
                "pred": [gather_pred_classes, gather_pred_frame],
                "target": [gather_target, target_class],
            }

            metric_dict = self.evaluate_metric(pred_target_dict)
            self.log("mAP", metric_dict["mAP"], on_epoch=True, prog_bar=True)
            self.log("multilabel f1", multilabel_f1_test, on_epoch=True, prog_bar=True)
            #self.log("multilabel f1 classes", multilabel_f1_test_classes, on_epoch=True, prog_bar = True)
            self.log("mAP test", mAP_test, on_epoch=True, prog_bar = True)
            #self.log("mAP test classes", mAP_test_classes, on_epoch=True, prog_bar = True)
            for i in range(10):
                self.log("mAP test class: " + config.id2classes[i] , mAP_test_classes[i], on_epoch=True, prog_bar = True)
                self.log("multilabel f1 class: " +config.id2classes[i] , multilabel_f1_test_classes[i], on_epoch=True, prog_bar = True)
            sedeval_metrics = log_sedeval_metrics(
            self.decoded_05_buffer, os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["val_tsv"],
        ))
            event_f1_macro = sedeval_metrics[0]
            event_f1 = sedeval_metrics[1]
            if isinstance(sedeval_metrics[2], dict):
                for keys, values in sedeval_metrics[2].items():
                    if keys == "class_wise":
                        for class_name, metrics in values.items():
                            class_wise_f1[class_name] = values[class_name]["f_measure"]["f_measure"]
                metric_dict["class_wise_f1"] = class_wise_f1
                #self.log("class_wise_f1", metric_dict["class_wise_f1"], on_epoch=True, prog_bar=False)
            self.log("event_f1_macro", event_f1_macro, on_epoch=True, prog_bar=True)
            self.log("event_f1", event_f1, on_epoch=True, prog_bar=True)
            """self.log(
                
                "mAUC",
                metric_dict["mAUC"],
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
            )"""
            """self.log(
                "dprime",
                metric_dict["dprime"],
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
            )"""
            # self.log("precision", metric_dict["precision"], on_epoch = True, prog_bar=True, sync_dist=False)
            # self.log("recall", metric_dict["recall"], on_epoch = True, prog_bar=True, sync_dist=False)
            #self.log(
            #    "f1", metric_dict["f1"], on_epoch=True, prog_bar=True, sync_dist=False
            #)
            # self.log.tb_writer.add_image('Confusion Matrix', metric_dict["confusion_mat"], global_step=self.global_step)
            # self.log("mAUC", metric_dict["mAUC"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            # self.log("dprime", metric_dict["dprime"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            self.multilabel_f1_test.reset()
            self.multilabel_f1_test_classes.reset()
            self.mAP_test.reset()
            self.mAP_test_classes.reset()
        else:
            self.log(
                "acc", metric_dict["acc"], on_epoch=True, prog_bar=True, sync_dist=False
            )
        # dist.barrier()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.05,
        )

        # Change: SWA, deprecated
        # optimizer = SWA(optimizer, swa_start=10, swa_freq=5)
        def lr_foo(epoch):
            if epoch < 3:
                # warm up lr
                lr_scale = self.config.lr_rate[epoch]
            else:
                # warmup schedule
                lr_pos = int(
                    -1 - bisect.bisect_left(self.config.lr_scheduler_epoch, epoch)
                )
                if lr_pos < -3:
                    lr_scale = max(self.config.lr_rate[0] * (0.98**epoch), 0.03)
                else:
                    lr_scale = self.config.lr_rate[lr_pos]
            return lr_scale

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)

        return [optimizer], [scheduler]


class Ensemble_SEDWrapper(pl.LightningModule):
    def __init__(self, sed_models, config, dataset):
        super().__init__()

        self.sed_models = nn.ModuleList(sed_models)
        self.config = config
        self.dataset = dataset

    def evaluate_metric(self, pred, ans):
        if self.config.dataset_type == "audioset":
            mAP = np.mean(average_precision_score(ans, pred, average=None))

            # mAUC = np.mean(roc_auc_score(ans, pred, average = None))
            # dprime = d_prime(mAUC)
            return {"mAP": mAP}  # , "mAUC": mAUC, "dprime": dprime}
        else:
            acc = accuracy_score(ans, np.argmax(pred, 1))
            return {"acc": acc}

    def forward(self, x, sed_index, mix_lambda=None):
        self.sed_models[sed_index].eval()
        preds = []
        pred_maps = []
        # time shifting optimization
        if self.config.fl_local or self.config.dataset_type != "audioset":
            shift_num = 1  # framewise localization cannot allow the time shifting
        else:
            shift_num = 10
        for i in range(shift_num):
            pred, pred_map = self.sed_models[sed_index](x)
            pred_maps.append(pred_map.unsqueeze(0))
            preds.append(pred.unsqueeze(0))
            x = self.time_shifting(x, shift_len=100 * (i + 1))
        preds = torch.cat(preds, dim=0)
        pred_maps = torch.cat(pred_maps, dim=0)
        pred = preds.mean(dim=0)
        pred_map = pred_maps.mean(dim=0)
        return pred, pred_map

    def time_shifting(self, x, shift_len):
        shift_len = int(shift_len)
        new_sample = torch.cat([x[:, shift_len:], x[:, :shift_len]], axis=1)
        return new_sample

    def test_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        if self.config.fl_local:
            pred = (
                torch.zeros(len(batch["waveform"]), self.config.classes_num)
                .float()
                .to(self.device_type)
            )
            pred_map = (
                torch.zeros(len(batch["waveform"]), 1024, self.config.classes_num)
                .float()
                .to(self.device_type)
            )
            for j in range(len(self.sed_models)):
                temp_pred, temp_pred_map = self(batch["waveform"], j)
                pred = pred + temp_pred
                pred_map = pred_map + temp_pred_map
            pred = pred / len(self.sed_models)
            pred_map = pred_map / len(self.sed_models)
            return [
                pred.detach().cpu().numpy(),
                pred_map.detach().cpu().numpy(),
                batch["audio_name"],
                batch["real_len"].cpu().numpy(),
            ]
        else:
            pred = (
                torch.zeros(len(batch["waveform"]), self.config.classes_num)
                .float()
                .to(self.device_type)
            )
            for j in range(len(self.sed_models)):
                temp_pred, _ = self(batch["waveform"], j)
                pred = pred + temp_pred
            pred = pred / len(self.sed_models)
            return [
                pred.detach(),
                batch["target_frames"].detach(),
            ]

    def test_epoch_end(self, test_step_outputs):
        self.device_type = next(self.parameters()).device
        if self.config.fl_local:
            pred = np.concatenate([d[0] for d in test_step_outputs], axis=0)
            pred_map = np.concatenate([d[1] for d in test_step_outputs], axis=0)
            audio_name = np.concatenate([d[2] for d in test_step_outputs], axis=0)
            real_len = np.concatenate([d[3] for d in test_step_outputs], axis=0)
            heatmap_file = os.path.join(
                self.config.heatmap_dir,
                self.config.test_file + "_" + str(self.device_type) + ".npy",
            )
            save_npy = [
                {
                    "audio_name": audio_name[i],
                    "heatmap": pred_map[i],
                    "pred": pred[i],
                    "real_len": real_len[i],
                }
                for i in range(len(pred))
            ]
            np.save(heatmap_file, save_npy)
        else:
            pred = torch.cat([d[0] for d in test_step_outputs], dim=0)
            target = torch.cat([d[1] for d in test_step_outputs], dim=0)
            gather_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
            gather_target = [
                torch.zeros_like(target) for _ in range(dist.get_world_size())
            ]

            dist.barrier()
            if self.config.dataset_type == "audioset":
                metric_dict = {
                    "mAP": 0.0,
                    "mAUC": 0.0,
                    "dprime": 0.0,
                    "f1": 0.0,
                }
            else:
                metric_dict = {"acc": 0.0}
            dist.all_gather(gather_pred, pred)
            dist.all_gather(gather_target, target)
            if dist.get_rank() == 0:
                gather_pred = torch.cat(gather_pred, dim=0).cpu().numpy()
                gather_target = torch.cat(gather_target, dim=0).cpu().numpy()
                if self.config.dataset_type == "scv2":
                    gather_target = np.argmax(gather_target, 1)
                metric_dict = self.evaluate_metric(gather_pred, gather_target)
                print(self.device_type, dist.get_world_size(), metric_dict, flush=True)
            if self.config.dataset_type == "audioset":
                self.log(
                    "mAP",
                    metric_dict["mAP"],
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
                
                """self.log(
                    "mAUC",
                    metric_dict["mAUC"],
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=False,
                )"""
                """self.log(
                    "dprime",
                    metric_dict["dprime"],
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=False,
                )"""
                # self.log("precision", metric_dict["precision"], on_epoch = True, prog_bar=True, sync_dist=False)
                # self.log("recall", metric_dict["recall"], on_epoch = True, prog_bar=True, sync_dist=False)
                """self.log(
                    "f1",
                    metric_dict["f1"],
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=False,
                )"""
                # self.log("mAUC", metric_dict["mAUC"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
                # self.log("dprime", metric_dict["dprime"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            else:
                self.log(
                    "acc",
                    metric_dict["acc"] * float(dist.get_world_size()),
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
            dist.barrier()
