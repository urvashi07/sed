import h5py
import numpy as np
import glob
import os
import argparse
import torch
from desed_task.dataio.datasets import read_audio
import yaml
import pandas as pd
#from desed_task.utils.download import download_from_url
from tqdm import tqdm
from pathlib import Path
import torchaudio

parser = argparse.ArgumentParser("Extract Embeddings with Audioset Pretrained Models")


class WavDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder,
        pad_to=10,
        fs=16000,
        feats_pipeline=None
    ):
        self.fs = fs
        self.pad_to = pad_to * fs if pad_to is not None else None
        self.examples = glob.glob(os.path.join(folder, "*.wav"))
        self.feats_pipeline = feats_pipeline

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        c_ex = self.examples[item]

        mixture, _, _, padded_indx = read_audio(
            c_ex,  False, False, self.pad_to
        )

        if self.feats_pipeline is not None:
            mixture = self.feats_pipeline(mixture)
        return mixture, Path(c_ex).stem


def extract(batch_size, folder, dset_name, torch_dset, embedding_model):

    Path(folder).mkdir(parents=True, exist_ok=True)
    f = h5py.File(os.path.join(folder, '{}.hdf5'.format(dset_name)), "w")
    if type(embedding_model).__name__ == "Cnn14_16k":
        emb_size = int(256*8)
    else:
        emb_size = 768
    global_embeddings = f.create_dataset('global_embeddings', (len(torch_dset), emb_size), dtype=np.float32)
    frame_embeddings = f.create_dataset('frame_embeddings', (len(torch_dset), emb_size, 496), dtype=np.float32)
    filenames_emb = f.create_dataset("filenames", data=["example_00.wav"] * len(torch_dset))
    dloader = torch.utils.data.DataLoader(torch_dset,
                                          batch_size=batch_size,
                                          drop_last=False
                                          )
    global_indx = 0
    for i, batch in enumerate(tqdm(dloader)):
        feats, filenames = batch
        feats = feats.cuda()

        with torch.inference_mode():
            emb = embedding_model(feats)
            c_glob_emb =  emb["global"]
            c_frame_emb = emb["frame"]
        # enumerate, convert to numpy and write to h5py
        bsz = feats.shape[0]
        for b_indx in range(bsz):
            global_embeddings[global_indx] = c_glob_emb[b_indx].detach().cpu().numpy()
            #global_embeddings.attrs[filenames[b_indx]] = global_indx
            frame_embeddings[global_indx] = c_frame_emb[b_indx].detach().cpu().numpy()
            #frame_embeddings.attrs[filenames[b_indx]] = global_indx
            filenames_emb[global_indx] = filenames[b_indx]
            global_indx += 1

if __name__ == "__main__":

    with open("/home/unegi2s/Documents/sed_github/sed/recipes/dcase2023_task4_baseline/confs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    output_dir = os.path.join("/work/unegi2s/embeddings/beats")
    # loading model
    feature_extraction = None # integrated in the model
        # use beats as additional feature
    from local.beats.BEATs import BEATsModel
    pretrained = BEATsModel(cfg_path="/work/unegi2s/beats/BEATS_iter3_plus_AS2M.pt")

    pretrained = pretrained.cuda()

    pretrained.eval()
    synth_df = pd.read_csv(os.path.join(config["data"]["prefix_folder"], config["data"]["synth_tsv"]), sep="\t")
    synth_set = WavDataset(
        os.path.join(config["data"]["prefix_folder"], config["data"]["synth_folder"]),
        feats_pipeline=feature_extraction)

    synth_set[0]

    strong_set = WavDataset(
        os.path.join(config["data"]["prefix_folder"], config["data"]["strong_folder"]),
        feats_pipeline=feature_extraction)
    
    
    weak_df = pd.read_csv(os.path.join(config["data"]["prefix_folder"], config["data"]["weak_tsv"]), sep="\t")
    train_weak_df = weak_df.sample(
        frac=config["training"]["weak_split"],
        random_state=config["training"]["seed"])

    valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
    train_weak_df = train_weak_df.reset_index(drop=True)
    weak_set = WavDataset(
        os.path.join(config["data"]["prefix_folder"], config["data"]["weak_folder"]),
        feats_pipeline=feature_extraction)

    unlabeled_set = WavDataset(
        os.path.join(config["data"]["prefix_folder"], config["data"]["unlabeled_folder"]),
        feats_pipeline=feature_extraction)

    synth_df_val = pd.read_csv(os.path.join(config["data"]["prefix_folder"], config["data"]["synth_val_tsv"]),
                               sep="\t")
    synth_val = WavDataset(
        os.path.join(config["data"]["prefix_folder"], config["data"]["synth_val_folder"]),
        feats_pipeline=feature_extraction
    )

    weak_val = WavDataset(
        os.path.join(config["data"]["prefix_folder"], config["data"]["weak_folder"]),
        feats_pipeline=feature_extraction
    )

    devtest_dataset = WavDataset(
        os.path.join(config["data"]["prefix_folder"], config["data"]["test_folder"]), feats_pipeline=feature_extraction)
    for k, elem in {"synth_train": synth_set, "weak_train": weak_set,
                    "strong_train": strong_set,
                    "unlabeled_train" : unlabeled_set,
                   "synth_val" : synth_val,
                   "weak_val" : weak_val,
                    "devtest": devtest_dataset}.items():
    #for k, elem in {"strong_train": strong_set}.items():
    #for k, elem in {"devtest": devtest_dataset}.items():
        extract(8, output_dir, k, elem, pretrained)
