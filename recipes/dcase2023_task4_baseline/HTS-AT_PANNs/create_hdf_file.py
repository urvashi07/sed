import os
import h5py
import numpy as np
import librosa
import yaml


def create_hdf_file(data, file_name, sr):

    # Create a new HDF5 file
    file = h5py.File(file_name, "w")
    groups = data.keys()

    for group_name in groups:
        group = file.create_group(group_name)

        subgroups = []

        if group_name == "train":
            subgroups = list(data["train"].keys()) # [synth, strong, weak]
        elif group_name == "eval":
            subgroups = list(data["eval"].keys())
        else:
            subgroups = data["test"].keys()
    
        for subgroup_name in subgroups:
            subgroup = group.create_group(subgroup_name)
            data_subgroup = data[group_name][subgroup_name]
            subgroup.attrs["folder_path"] = os.path.dirname(data_subgroup[0].filepath)
        
            for element in data_subgroup:
                subsubgroup = subgroup.create_group(element.filename)
                # Set attributes in the subgroup
                subsubgroup.attrs["class_labels"] = element.class_labels
                subsubgroup.attrs["onset_times"] = element.onset_times
                subsubgroup.attrs["offset_times"] = element.offset_times

                # Load audio file and create a dataset
                waveform, sr = librosa.load(element.filepath, sr = sr)
                subsubgroup.create_dataset("waveform", data=waveform)
                subsubgroup.attrs["sr"] = sr
    # Close the file
    file.close()