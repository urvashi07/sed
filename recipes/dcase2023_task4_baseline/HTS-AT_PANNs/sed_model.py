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
from torchmetrics.classification import MulticlassConfusionMatrix
import config

from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
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
        print(self.device)
        self.get_weak_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
            len(self.config.classes2id), average="macro"
        ).to(self.device)
        #self.df_eval = df_eval
        #self.df_test = df_test
        self.prefix_folder = prefix_folder
        #self.filename_eval = self.df_eval.filename.unique()
        #self.filename_test = self.df_test.filename.unique()
        self.prediction_dfs = []
        conf_file_path = "/home/unegi/Documents/dcase2023_task4_gitlab/DESED_task/recipes/dcase2023_task4_baseline/confs/default.yaml"
        with open(conf_file_path, "r") as f:
            self.configs = yaml.safe_load(f)
        # for weak labels we simply compute f1 score
        self.val_decoded_pred = pd.DataFrame(columns=["filename", "event_label", "onset", "offset"])
        self.test_decoded_pred = pd.DataFrame(columns=["filename", "event_label", "onset", "offset"])
    

    def evaluate_metric(self, pred_target_dict):
        ap = []
        pred = pred_target_dict["pred"]
        target = pred_target_dict["target"]

        pred_class = pred[0]
        pred_frame = pred[1]
        target_class = target[1]
        target = target[0]
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

            reshaped_pred_frame = pred_frame.transpose(0, 2, 1)
            reshaped_target = target.transpose(0, 2, 1)
            # reshaped_pred_class = pred_class.transpose(1,0)
            # reshaped_target_class = target_class.transpose(1,0)

            threshold = 0.5
            predicted_labels = pred_class > threshold
            #reshaped_target_class = target
            ap_values = []
            auc_scores = []
            f1_scores = []
            for class_idx in range(10):
                class_target_frame = reshaped_target[:, class_idx, :]
                class_pred_frame = reshaped_pred_frame[:, class_idx, :]
                ap = average_precision_score(
                    class_target_frame, class_pred_frame, average=None
                )
                ap_values.append(ap)
                # auc = roc_auc_score(class_ans, class_pred)
                # auc_scores.append(auc)

                # f1 = f1_score(reshaped_target_class[class_idx, :], predicted_labels[class_idx, :], task = "multiclass")
                # f1_scores.append(f1)

            mAP = np.mean(ap_values)
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
        indx_synth, indx_weak = [4,2]
        self.device_type = next(self.parameters()).device
        loss_frame = 0
        loss_class = 0
        if isinstance(batch, list):
            indx_synth, indx_weak = [4,2]
            print(batch)
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
            print(pred_clip.shape)
            print(pred_frame.shape)
        # loss = self.loss_func(pred_clip, batch["target_frames"])
        # pred_frame = pred_frame.float()

        

            target = batch["target"].float()
        #target_classes = batch["target_classes"]
            loss_frame = self.frame_loss_func(pred_frame[strong_mask], target[strong_mask])
            loss_class = self.class_loss_func(pred_clip[weak_mask], target[weak_mask])
            loss = loss_frame + loss_class
            print("................")
            print(loss)
            self.log("train_loss", loss, on_epoch=True, prog_bar=True)
            return loss

    # def training_epoch_end(self, outputs):
    # Change: SWA, deprecated
    # for opt in self.trainer.optimizers:
    #     if not type(opt) is SWA:
    #         continue
    #     opt.swap_swa_sgd()
    # self.dataset.generate_queue()

    def validation_step(self, batch, batch_idx):
        audio_name, audio, label = batch["audio_name"], batch["waveform"], batch["target"]
        indx_synth, indx_weak = [6,6]
        batch_num = len(batch)
        pred_clip, pred_frame = self(audio)
        strong_mask = torch.zeros(batch_num).bool()
        weak_mask = torch.zeros(batch_num).bool()
        strong_mask[:indx_synth] = 1
        weak_mask[indx_synth : indx_weak + indx_synth] = 1
        print("validation prediction")
        target = label.float()
        #target_classes = batch["target_classes"].float()
        loss_frame = self.frame_loss_func(pred_frame, target)
        #loss_class = self.class_loss_func(pred_clip, target_classes)
        loss = loss_frame# + loss_class
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
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
        #print(mask_weak)
        #print(pred_clip[mask_weak])
        # accumulate f1 score for weak labels
        if not pred_clip[mask_weak].numel() == 0:
            self.get_weak_f1_seg_macro(
                pred_clip[weak_mask], target[weak_mask].long()
            )

        filenames_synth = [
                x
                for x in audio_name
                if Path(x).parent == Path(os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["synth_val_folder"]))
            ]
        decoded_strong = batched_decode_preds(
        pred_frame, filenames_synth, thresholds=0.5, median_filter=7, pad_indx=None,
        hop_length = self.configs["feats"]["hop_length"], sr = self.configs["feats"]["sample_rate"]
        )
        self.val_decoded_pred = pd.concat([self.val_decoded_pred, decoded_strong], 
                                          ignore_index = True)


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
        self.device_type = next(self.parameters()).device

        # print(target.shape)
        pred_classes = torch.cat([d[0] for d in validation_step_outputs], dim=0)
        pred_frame = torch.cat([d[1] for d in validation_step_outputs], dim=0)
        target = torch.cat([d[2] for d in validation_step_outputs], dim=0)
        #target_frame = torch.cat([d[3] for d in validation_step_outputs], dim=0)

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

            for keys, values in sedeval_metrics[2].items():
                if keys == "class_wise":
                    for class_name, metrics in values.items():
                        class_wise_f1[class_name] = values[class_name]["f_measure"]["f_measure"]
            if target.dim()> 2:
                target_class = torch.any(target == 1, dim=1).int()
            else:
                target_class = target
            
            
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
            "target": [gather_target, target_class],
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
            self.log("class_wise_f1", metric_dict["class_wise_f1"], on_epoch=True, prog_bar=True)
            self.log("event_f1_macro", synth_event_f1_macro, on_epoch=True, prog_bar=True)
            self.log("event_f1", synth_event_f1_macro, on_epoch=True, prog_bar=True)
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
            """self.log("class-wise-f_measure", class_wise_f_measure, 
                on_epoch=True,
                prog_bar=True,
                sync_dist=False)
            self.log("overall-wise-f_measure", overall_wise_f_measure, 
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,)
            ##self.df_eval["event_label"] = self.df_eval["event_label"].map(config.classes2id)"""
            
            self.prediction_dfs = []
            """df_cm = pd.DataFrame(metric_dict["confusion_matrix"].numpy(), index = range(10), columns=range(10))
                plt.figure(figsize = (10,7))
                fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
                plt.close(fig_)
        
                #self.log.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)"""
        else:
            self.log(
                "acc", metric_dict["acc"], on_epoch=True, prog_bar=True, sync_dist=False
            )
        self.get_weak_f1_seg_macro.reset()
        metric_dict = {"mAP": 0.0}

    def time_shifting(self, x, shift_len):
        shift_len = int(shift_len)
        new_sample = torch.cat([x[:, shift_len:], x[:, :shift_len]], axis=1)
        return new_sample

    def test_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        preds = []
        # time shifting optimization
        if self.config.fl_local or self.config.dataset_type != "audioset":
            shift_num = 1  # framewise localization cannot allow the time shifting
        else:
            shift_num = 10
        for i in range(shift_num):
            pred_clip, pred_frame = self(batch["waveform"])
            # preds.append(pred_clip.unsqueeze(0))
            preds.append(pred_frame)
            batch["waveform"] = self.time_shifting(
                batch["waveform"], shift_len=100 * (i + 1)
            )
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
        
        decoded_strong = batched_decode_preds(
        pred_frame, filenames_test, thresholds=0.5, median_filter=7, pad_indx=None,
        hop_length = self.configs["feats"]["hop_length"], sr = self.configs["feats"]["sample_rate"]
        )
        self.test_decoded_pred = pd.concat([self.test_decoded_pred, decoded_strong], 
                                          ignore_index = True)
        
        

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
        self.device_type = next(self.parameters()).device
        # pred = torch.cat([d[0] for d in test_step_outputs], dim = 0)
        # target = torch.cat([d[1] for d in test_step_outputs], dim = 0)

        pred_classes = torch.cat([d[0] for d in test_step_outputs], dim=0)
        pred_frame = torch.cat([d[1] for d in test_step_outputs], dim=0)
        target = torch.cat([d[2] for d in test_step_outputs], dim=0)
        target_class = torch.any(target == 1, dim=1).int()
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

            sedeval_metrics = log_sedeval_metrics(
            self.test_decoded_pred, os.path.join(self.configs["data"]["prefix_folder"], self.configs["data"]["val_tsv"],
        ))
            event_f1_macro = sedeval_metrics[0]
            event_f1 = sedeval_metrics[1]
            for keys, values in sedeval_metrics[2].items():
                if keys == "class_wise":
                    for class_name, metrics in values.items():
                        class_wise_f1[class_name] = values[class_name]["f_measure"]["f_measure"]
            metric_dict["class_wise_f1"] = class_wise_f1
            self.log("class_wise_f1", metric_dict["class_wise_f1"], on_epoch=True, prog_bar=True)
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
