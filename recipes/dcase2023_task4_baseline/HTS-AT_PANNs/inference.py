import pandas as pd
import scipy
from pathlib import Path
import numpy as np
import config
import sed_eval
import os
from sed_scores_eval.utils.scores import create_score_dataframe
from dcase_util.data import DecisionEncoder

def get_event_list_current_file(df, fname):
    """
    Get list of events for a given filename
    Args:
        df: pd.DataFrame, the dataframe to search on
        fname: the filename to extract the value from the dataframe
    Returns:
         list of events (dictionaries) for the given filename
    """
    event_file = df[df["filename"] == fname]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict("records")
    else:
        event_list_for_current_file = event_file.to_dict("records")

    return event_list_for_current_file

def event_based_evaluation_df(
    reference, estimated, t_collar=0.200, percentage_of_length=0.2
):
    """ Calculate EventBasedMetric given a reference and estimated dataframe

    Args:
        reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            reference events
        estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            estimated events to be compared with reference
        t_collar: float, in seconds, the number of time allowed on onsets and offsets
        percentage_of_length: float, between 0 and 1, the percentage of length of the file allowed on the offset
    Returns:
         sed_eval.sound_event.EventBasedMetrics with the scores
    """

    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling="zero_score",
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return event_based_metric

def segment_based_evaluation_df(reference, estimated, time_resolution=1.0):
    """ Calculate SegmentBasedMetrics given a reference and estimated dataframe

        Args:
            reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
                reference events
            estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
                estimated events to be compared with reference
            time_resolution: float, the time resolution of the segment based metric
        Returns:
             sed_eval.sound_event.SegmentBasedMetrics with the scores
        """
    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes, time_resolution=time_resolution
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        segment_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return segment_based_metric

def compute_sed_eval_metrics(predictions, groundtruth):
    """ Compute sed_eval metrics event based and segment based with default parameters used in the task.
    Args:
        predictions: pd.DataFrame, predictions dataframe
        groundtruth: pd.DataFrame, groundtruth dataframe
    Returns:
        tuple, (sed_eval.sound_event.EventBasedMetrics, sed_eval.sound_event.SegmentBasedMetrics)
    """
    metric_event = event_based_evaluation_df(
        groundtruth, predictions, t_collar=0.200, percentage_of_length=0.2
    )
    metric_segment = segment_based_evaluation_df(
        groundtruth, predictions, time_resolution=1.0
    )

    return metric_event, metric_segment

def log_sedeval_metrics(predictions, ground_truth, save_dir=None):
    """ Return the set of metrics from sed_eval
    Args:
        predictions: pd.DataFrame, the dataframe of predictions.
        ground_truth: pd.DataFrame, the dataframe of groundtruth.
        save_dir: str, path to the folder where to save the event and segment based metrics outputs.

    Returns:
        tuple, event-based macro-F1 and micro-F1, segment-based macro-F1 and micro-F1
    """
    if isinstance(predictions, dict):
        if len(predictions) == 0:
            return 0.0, 0.0, 0.0, 0.0
    elif predictions.empty:
        return 0.0, 0.0, 0.0, 0.0

    gt = pd.read_csv(ground_truth, sep="\t")

    event_res, segment_res = compute_sed_eval_metrics(predictions, gt)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
            f.write(str(event_res))

        with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
            f.write(str(segment_res))

    return (
        event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        event_res.results()["overall"]["f_measure"]["f_measure"],
        event_res.results(),
        segment_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        segment_res.results()["overall"]["f_measure"]["f_measure"],
    )

def find_contiguous_regions(activity_array):
        """Find contiguous regions from bool valued numpy.array.
        Transforms boolean values for each frame into pairs of onsets and offsets.

        Parameters
        ----------
        activity_array : numpy.array [shape=(t)]
            Event activity array, bool values

        Returns
        -------
        numpy.ndarray [shape=(2, number of found changes)]
            Onset and offset indices pairs in matrix

        """

        # Find the changes in the activity_array
        change_indices = np.logical_xor(activity_array[1:], activity_array[:-1]).nonzero()[0]

        # Shift change_index with one, focus on frame after the change.
        change_indices += 1

        if activity_array[0]:
            # If the first element of activity_array is True add 0 at the beginning
            change_indices = np.r_[0, change_indices]

        if activity_array[-1]:
            # If the last element of activity_array is True, add the length of the array
            change_indices = np.r_[change_indices, activity_array.size]

        # Reshape the result into two columns
        return change_indices.reshape((-1, 2))

def _frame_to_time(frame, hop_length, sr):
        return (frame * hop_length) / sr

def decode_strong(labels, hop_length, sr):
    """ Decode the encoded strong labels
        Args:
            labels: numpy.array, the encoded labels to be decoded
        Returns:
            list
            Decoded labels, list of list: [[label, onset offset], ...]
     """
    result_labels = []
    for i, label_column in enumerate(labels.T):
        change_indices = DecisionEncoder().find_contiguous_regions(label_column)

        # append [label, onset, offset] in the result list
        for row in change_indices:
            result_labels.append(
                [
                 list(config.classes2id.keys())[i],
                 _frame_to_time(row[0], hop_length, sr),
                 _frame_to_time(row[1], hop_length, sr),
                ]
              )
    return result_labels


def batched_decode_preds(
            strong_preds, filenames, hop_length, sr, thresholds=[0.5], median_filter=7, pad_indx=None,
            ):
    """ Decode a batch of predictions to dataframes. Each threshold gives a different dataframe and stored in a
            dictionary

        Args:
                strong_preds: torch.Tensor, batch of strong predictions.
                filenames: list, the list of filenames of the current batch.
                encoder: ManyHotEncoder object, object used to decode predictions.
                thresholds: list, the list of thresholds to be used for predictions.
                median_filter: int, the number of frames for which to apply median window (smoothing).
                pad_indx: list, the list of indexes which have been used for padding.

        Returns:
                dict of predictions, each keys is a threshold and the value is the DataFrame of predictions.
    """
    scores_raw = {}
    scores_postprocessed = {}
    prediction_dfs = {}
    labels = list(config.classes2id.keys())
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()
    for j in range(strong_preds.shape[0]):  # over batches
        audio_id = Path(filenames[j]).stem
        filename = audio_id + ".wav"
        c_scores = strong_preds[j]
        if pad_indx is not None:
            true_len = int(c_scores.shape[-1] * pad_indx[j].item())
            c_scores = c_scores[:true_len]
        c_scores = c_scores.detach().cpu().numpy()
        scores_raw[audio_id] = create_score_dataframe(scores=c_scores,
                                                    timestamps=_frame_to_time(np.arange(len(c_scores)+1), hop_length, sr),
                                                    event_classes=labels,
                                                                            )
        c_scores = scipy.ndimage.filters.median_filter(c_scores, (median_filter, 1))
        scores_postprocessed[audio_id] = create_score_dataframe(scores=c_scores,
                                                                timestamps=_frame_to_time(np.arange(len(c_scores)+1), hop_length, sr),
                                                                event_classes=labels,
                                                                            )
        for c_th in thresholds:
            pred = c_scores > c_th
            pred = decode_strong(pred, hop_length, sr)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = filename
            prediction_dfs[c_th] = pd.concat([prediction_dfs[c_th], pred], ignore_index=True)
    return scores_raw, scores_postprocessed, prediction_dfs

