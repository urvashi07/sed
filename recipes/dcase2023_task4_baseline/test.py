import os
import pandas as pd
from codecarbon import OfflineEmissionsTracker
import torch
import torch.nn as nn
import numpy as np
from local.utils import (
    batched_decode_preds,
    log_sedeval_metrics,
)
import sed_scores_eval
from torchaudio.transforms import AmplitudeToDB
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
    compute_psds_from_scores
)

class TestModel(nn.Module):
    def __init__(self, mel_spec, sed_student, sed_teacher, scaler, supervised_loss, hparams, encoder, evaluation, log, fast_dev_run, exp_dir) -> None:
        super(TestModel, self).__init__()
        self.mel_spec = mel_spec
        self.sed_student = sed_student
        self.sed_teacher = sed_teacher
        self.scaler = scaler
        self.supervised_loss = supervised_loss
        self.hparams = hparams
        self.encoder = encoder
        self.evaluation = evaluation
        self.fast_dev_run = fast_dev_run
        self.logger = None
        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer_student = {k: pd.DataFrame() for k in test_thresholds}
        self.test_psds_buffer_teacher = {k: pd.DataFrame() for k in test_thresholds}

        self.decoded_student_05_buffer = pd.DataFrame()
        self.decoded_teacher_05_buffer = pd.DataFrame()
        self.test_scores_raw_buffer_student = {}
        self.test_scores_raw_buffer_teacher = {}
        self.test_scores_postprocessed_buffer_student = {}
        self.test_scores_postprocessed_buffer_teacher = {}

        if exp_dir is not None:
            self.exp_dir = exp_dir
        else:
            self.exp_dir = self.hparams["log_dir"]

    def take_log(self, mels):
        """ Apply the log transformation to mel spectrograms.
        Args:
            mels: torch.Tensor, mel spectrograms for which to apply log.

        Returns:
            Tensor: logarithmic mel spectrogram of the mel spectrogram given as input
        """

        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

    def detect(self, mel_feats, model):
        return model(self.scaler(self.take_log(mel_feats)))
    
    

    def test_start_tracking(self):
        if self.evaluation:
            os.makedirs(
                os.path.join(self.exp_dir, "evaluation_codecarbon"), exist_ok=True
            )
            self.tracker_eval = OfflineEmissionsTracker(
                "DCASE Task 4 SED EVALUATION",
                output_dir=os.path.join(self.exp_dir, "evaluation_codecarbon"),
                log_level="warning",
                country_iso_code="FRA",
            )
            self.tracker_eval.start()
        else:
            os.makedirs(os.path.join(self.exp_dir, "devtest_codecarbon"), exist_ok=True)
            self.tracker_devtest = OfflineEmissionsTracker(
                "DCASE Task 4 SED DEVTEST",
                output_dir=os.path.join(self.exp_dir, "devtest_codecarbon"),
                log_level="warning",
                country_iso_code="FRA",
            )
            self.tracker_devtest.start()
    
    def test_step(self, batch, ):

        audio, labels, padded_indxs, filenames = batch        
        
        # prediction for student
         
        mels = self.mel_spec(audio)
    
        # Set the model to evaluation mode
        self.sed_student.eval()
        self.sed_teacher.eval()

        # Forward pass
        with torch.no_grad():     
        
            # prediction for student
            strong_preds_student, weak_preds_student = self.detect(mels, self.sed_student)
            # prediction for teacher
            strong_preds_teacher, weak_preds_teacher = self.detect(mels, self.sed_teacher)
        
            if not self.evaluation:
                loss_strong_student = self.supervised_loss(strong_preds_student, labels)
                loss_strong_teacher = self.supervised_loss(strong_preds_teacher, labels)

                #self.log("test/student/loss_strong", loss_strong_student)
                #self.log("test/teacher/loss_strong", loss_strong_teacher)

                # compute psds
            (
            scores_raw_student_strong, scores_postprocessed_student_strong,
            decoded_student_strong,
            ) = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_student.keys()) + [.5],
            )

            self.test_scores_raw_buffer_student.update(scores_raw_student_strong)
            self.test_scores_postprocessed_buffer_student.update(
            scores_postprocessed_student_strong
            )
            for th in self.test_psds_buffer_student.keys():
                self.test_psds_buffer_student[th] = pd.concat([self.test_psds_buffer_student[th], decoded_student_strong[th]], ignore_index=True)

            (
            scores_raw_teacher_strong, scores_postprocessed_teacher_strong,
            decoded_teacher_strong,
            ) = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_teacher.keys()) + [.5],
            )

            self.test_scores_raw_buffer_teacher.update(scores_raw_teacher_strong)
            self.test_scores_postprocessed_buffer_teacher.update(
            scores_postprocessed_teacher_strong
        )
            for th in self.test_psds_buffer_teacher.keys():
                self.test_psds_buffer_teacher[th] = pd.concat([self.test_psds_buffer_teacher[th], decoded_teacher_strong[th]], ignore_index=True)

            # compute f1 score
            self.decoded_student_05_buffer = pd.concat([self.decoded_student_05_buffer, decoded_student_strong[0.5]])
            self.decoded_teacher_05_buffer = pd.concat([self.decoded_teacher_05_buffer, decoded_teacher_strong[0.5]])

    def on_test_epoch_end(self):
        # pub eval dataset
        save_dir = os.path.join(self.exp_dir, "metrics_test")

        if self.evaluation:
            # only save prediction scores
            save_dir_student_raw = os.path.join(save_dir, "student_scores", "raw")
            sed_scores_eval.io.write_sed_scores(self.test_scores_raw_buffer_student, save_dir_student_raw)
            print(f"\nRaw scores for student saved in: {save_dir_student_raw}")

            save_dir_student_postprocessed = os.path.join(save_dir, "student_scores", "postprocessed")
            sed_scores_eval.io.write_sed_scores(self.test_scores_postprocessed_buffer_student, save_dir_student_postprocessed)
            print(f"\nPostprocessed scores for student saved in: {save_dir_student_postprocessed}")

            save_dir_teacher_raw = os.path.join(save_dir, "teacher_scores", "raw")
            sed_scores_eval.io.write_sed_scores(self.test_scores_raw_buffer_teacher, save_dir_teacher_raw)
            print(f"\nRaw scores for teacher saved in: {save_dir_teacher_raw}")

            save_dir_teacher_postprocessed = os.path.join(save_dir, "teacher_scores", "postprocessed")
            sed_scores_eval.io.write_sed_scores(self.test_scores_postprocessed_buffer_teacher, save_dir_teacher_postprocessed)
            print(f"\nPostprocessed scores for teacher saved in: {save_dir_teacher_postprocessed}")

            self.tracker_eval.stop()
            eval_kwh = self.tracker_eval._total_energy.kWh
            results = {"/eval/tot_energy_kWh": torch.tensor(float(eval_kwh))}
            with open(os.path.join(self.exp_dir, "evaluation_codecarbon", "eval_tot_kwh.txt"), "w") as f:
                f.write(str(eval_kwh))
        else:
            # calculate the metrics
            ground_truth = sed_scores_eval.io.read_ground_truth_events(self.hparams["data"]["test_tsv"])
            audio_durations = sed_scores_eval.io.read_audio_durations(self.hparams["data"]["test_dur"])
            if self.fast_dev_run:
                ground_truth = {
                    audio_id: ground_truth[audio_id]
                    for audio_id in self.test_scores_postprocessed_buffer_student
                }
                audio_durations = {
                    audio_id: audio_durations[audio_id]
                    for audio_id in self.test_scores_postprocessed_buffer_student
                }
            else:
                # drop audios without events
                ground_truth = {
                    audio_id: gt for audio_id, gt in ground_truth.items()
                    if len(gt) > 0
                }
                audio_durations = {
                    audio_id: audio_durations[audio_id]
                    for audio_id in ground_truth.keys()
                }
            psds1_student_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )
            psds1_student_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_student,
                ground_truth,
                audio_durations,
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=None,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )

            psds2_student_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )
            psds2_student_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_student,
                ground_truth,
                audio_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )

            psds1_teacher_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )
            psds1_teacher_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_teacher,
                ground_truth,
                audio_durations,
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=None,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )

            psds2_teacher_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )
            psds2_teacher_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_student,
                ground_truth,
                audio_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )

            event_macro_student = log_sedeval_metrics(
                self.decoded_student_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "student"),
            )[0]

            event_macro_teacher = log_sedeval_metrics(
                self.decoded_teacher_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "teacher"),
            )[0]

            # synth dataset
            intersection_f1_macro_student = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_student_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            # synth dataset
            intersection_f1_macro_teacher = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_teacher_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            best_test_result = torch.tensor(max(psds1_student_psds_eval, psds2_student_psds_eval))

            results = {
                "hp_metric": best_test_result,
                "test/student/psds1_psds_eval": psds1_student_psds_eval,
                "test/student/psds1_sed_scores_eval": psds1_student_sed_scores_eval,
                "test/student/psds2_psds_eval": psds2_student_psds_eval,
                "test/student/psds2_sed_scores_eval": psds2_student_sed_scores_eval,
                "test/teacher/psds1_psds_eval": psds1_teacher_psds_eval,
                "test/teacher/psds1_sed_scores_eval": psds1_teacher_sed_scores_eval,
                "test/teacher/psds2_psds_eval": psds2_teacher_psds_eval,
                "test/teacher/psds2_sed_scores_eval": psds2_teacher_sed_scores_eval,
                "test/student/event_f1_macro": event_macro_student,
                "test/student/intersection_f1_macro": intersection_f1_macro_student,
                "test/teacher/event_f1_macro": event_macro_teacher,
                "test/teacher/intersection_f1_macro": intersection_f1_macro_teacher,
            }
            self.tracker_devtest.stop()
            eval_kwh = self.tracker_devtest._total_energy.kWh
            results.update({"/test/tot_energy_kWh": torch.tensor(float(eval_kwh))})
            with open(os.path.join(self.exp_dir, "devtest_codecarbon", "devtest_tot_kwh.txt"), "w") as f:
                f.write(str(eval_kwh))

        if self.logger is not None:
            self.logger.log_metrics(results)
            self.logger.log_hyperparams(self.hparams, results)


        #for key in results.keys():
            #self.log(key, results[key], prog_bar=True, logger=True)
