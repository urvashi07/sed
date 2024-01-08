import numpy as np
import os
from numpy import floating, int16, number, int32, float32
from numpy.typing import NDArray
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import config
import yaml
import sys


@dataclass
class Event_Dataclass:
    filename: str
    onset_times: NDArray[floating]
    offset_times: NDArray[floating]
    class_labels: NDArray[int32]
    filepath: Path

def replace_labels_with_integers(label_string):
    labels = label_string.split(',')
    
    for label in labels:
        integer_labels = [config.classes2id[label] for label in labels if label in config.classes2id]
    return integer_labels

def read_csv(file_path):
    df = pd.read_csv(file_path, sep="\t")
    df = df.dropna()
    if "event_label" in df.columns:
        df["event_label"] = df["event_label"].map(config.classes2id)
    elif "event_labels" in df.columns:
        for index, row in df.iterrows():
            events = row["event_labels"]
        df['event_labels'] = df['event_labels'].apply(replace_labels_with_integers)

    return df


def get_file_info(df):
    data_dict = {}
    result = df.groupby("filename").agg(lambda x: x.tolist()).reset_index()

    # Convert the DataFrame to a dictionary with 'values' orientation
    data_dict = result.set_index("filename").T.to_dict("list")
    return data_dict


def convert_to_list_dataclass(event_dict, audio_dir_path, data_type):
    audio_data_list = []
    for filename, data in event_dict.items():
        if data_type == "strong":
            onset_times, offset_times, class_labels = data
            audio_data = Event_Dataclass(
                filename=filename,
                onset_times=np.array(onset_times),
                offset_times=np.array(offset_times),
                class_labels=np.array(class_labels),
                filepath=os.path.join(audio_dir_path, filename),
            )
        elif data_type == "weak":
            class_labels = data
            audio_data = Event_Dataclass(
                filename=filename,
                class_labels=np.array(class_labels),
                onset_times=np.array([]),  # Placeholder for empty arrays
                offset_times=np.array([]),
                filepath=os.path.join(audio_dir_path, filename),
            )
        audio_data_list.append(audio_data)
    return audio_data_list


def prepare_data(path, audio_path, data_type, data):
    df = read_csv(path)
    
    event_dict = get_file_info(df)
    list_audio_info = convert_to_list_dataclass(event_dict, audio_path, data_type)
    """if data == "eval-strong":
        print(df)
        print(list_audio_info[:10])
        sys.exit()"""
    return list_audio_info


def prepare_all_data(config):
    #*********************train data*******************
    list_audio_info_strong = prepare_data(
        path=os.path.join(config["prefix_folder"], config["strong_tsv"]),
        audio_path=os.path.join(config["prefix_folder"], config["strong_folder"]),
        data_type="strong",
        data = "train-strong"
    )
    list_audio_info_synth = prepare_data(
        path=os.path.join(config["prefix_folder"], config["synth_tsv"]),
        audio_path=os.path.join(config["prefix_folder"], config["synth_folder"]),
        data_type="strong",
        data = "train-synth"
    )

    list_audio_info_weak = prepare_data(
        path=os.path.join(config["prefix_folder"], config["weak_tsv"]),
        audio_path=os.path.join(config["prefix_folder"], config["weak_folder"]),
        data_type="weak",
        data = "train-weak"
    )


    #******************eval data*******************
    list_audio_info_eval_strong = prepare_data(
        path=os.path.join(config["prefix_folder"], config["val_tsv"]),
        audio_path=os.path.join(config["prefix_folder"], config["val_folder"]),
        data_type="strong",
        data = "eval-strong"
    )
    list_audio_info_eval_synth = prepare_data(
        path=os.path.join(config["prefix_folder"], config["synth_val_tsv"]),
        audio_path=os.path.join(config["prefix_folder"], config["synth_val_folder"]),
        data_type="strong",
        data = "eval-synth"
    )

    list_audio_info_eval_weak = prepare_data(
        path=os.path.join(config["prefix_folder"], config["weak_tsv"]),
        audio_path=os.path.join(config["prefix_folder"], config["weak_folder"]),
        data_type="weak",
        data = "eval-weak"
    )

    return {
        "train": {
            "strong": list_audio_info_strong,
            "synth": list_audio_info_synth,
            "weak" : list_audio_info_weak
        },
        "eval": {
            "strong": list_audio_info_eval_strong,
            "synth": list_audio_info_eval_synth,
            "weak": list_audio_info_eval_weak
        },
    }


if __name__ == "__main__":
    conf_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "confs/default.yaml",
    )

    with open(conf_file_path, "r") as f:
        configs = yaml.safe_load(f)

    all_data = prepare_all_data(configs["data"])
