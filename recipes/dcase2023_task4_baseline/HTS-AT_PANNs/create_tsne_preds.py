import numpy as np
import os
import pandas as pd
from pathlib import Path
import config
import yaml
import sys
from datasets_strong import SEDDataset_Strong
from torch.utils.data import Dataset, DataLoader, sampler, Subset, ConcatDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from model.htsat import HTSAT_Swin_Transformer
import h5py
from panns_models import *
import os
import torch
from sed_model import SEDWrapper
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv
import umap

from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str, help="Model to be used: panns, hts-at", default="panns"
    )
    args = parser.parse_args()
    print(args.model)

    args.model = args.model.lower()

    conf_file_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "confs/default.yaml",
                                )
    with open(conf_file_path, "r") as f:
        configs = yaml.safe_load(f)

    hdf_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), configs["hdf_file"])

    h5py_file = h5py.File(hdf_file_path, "r")

    test_data = h5py_file["test"]

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
    print("N_MELS")
    print(N_MELS)

    LEARNING_RATE = configs["opt"]["lr"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = SEDDataset_Strong(
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        data=test_data["strong"],
        config=configs,
        device=device,
    )
    len_dataset = len(test_dataset)

    #subset = torch.utils.Subset(test_dataset, np.arange(5))
    ckpt_path_epoch = str(100) 
    test_dataloader = DataLoader(
            dataset=test_dataset,
            num_workers=configs["training"]["num_workers"],
            batch_size=configs["training"]["batch_size"],
            shuffle=False,
            sampler=None,
        )

    if args.model == "panns":
        log_dir = os.path.join("./logs", "panns_all_data")
        model_config = {
            "sample_rate": SAMPLE_RATE,
            "window_size": WIN_LENGTH,
            "hop_size": HOP_LENGTH,
            "mel_bins": 64,
            "fmin": F_MIN,
            "fmax": F_MAX,
            "classes_num": 10,
        }
        sed_model = PANNsCNN14Att(**model_config)
        # weights = torch.load("Cnn14_DecisionLevelAtt_mAP0.425.pth", map_location = "cpu")
        # Fixed in V3
        # model.load_state_dict(weights["model"])
        sed_model.att_block = AttBlock(2048, 10, activation="sigmoid")
        
        #ckpt_path = configs["data"]["ckpt_panns_25"]
        ckpt_path = configs["data"]["ckpt_panns_"+ckpt_path_epoch]

        
    elif args.model == "hts-at" or args.model == "htsat":
        log_dir = os.path.join("./logs", "hts-at_all_data")

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

        ckpt_path = configs["data"]["ckpt_htsat_"+ckpt_path_epoch]
        #ckpt_path = configs["data"]["ckpt_htsat_50"]
        #ckpt_path = configs["data"]["ckpt_htsat_100"]

    tsne_dir = os.path.join(log_dir, "tsne", "pred", "epochs_" +ckpt_path_epoch)

    if not os.path.exists(tsne_dir):
        os.mkdir(tsne_dir)
        
    ckpt = torch.load(ckpt_path)
    model = SEDWrapper(
        sed_model=sed_model,
        config=config,
        prefix_folder=configs["data"]["prefix_folder"],
    )

    model.load_state_dict(ckpt["state_dict"], strict=False)

    if args.model == "panns":
        desired_layer = model.sed_model.fc1
    elif args.model == "hts-at" or args.model == "htsat":
        desired_layer = model.sed_model.tscam_conv

    embeddings = {}
    embeddings["panns"] = torch.Tensor()
    embeddings["htsat"] = torch.Tensor()
    labels_frame2class_all = torch.Tensor()
    pred_clip_all = torch.Tensor()
    labels_class = []

    def hook(module, input, output):
        global embeddings
        if args.model == "panns":
            embeddings["panns"] = torch.cat((embeddings["panns"], output), dim=0)
        elif args.model == "hts-at" or args.model == "htsat":
            embeddings["htsat"] = torch.cat((embeddings["htsat"], output), dim=0)

    hook_handle = desired_layer.register_forward_hook(hook)
    perplexities = [2, 25, 50, 100, 200]
    iterations = [250, 500, 1000, 2000, 5000]
    
    with torch.no_grad():
        model.eval()
        for batch in test_dataloader:
            pred_clip, pred_frame = model(batch["waveform"])
            #pred_clip = outputs["clipwise_output"]
            labels_frame2class = torch.any(batch["target"] == 1, dim=1).int().squeeze()
            if len(labels_frame2class.shape) < 2:
                labels_frame2class = labels_frame2class.unsqueeze(0)
            if len(pred_clip.shape) < 2:#added
                pred_clip = pred_clip.unsqueeze(0)#added
            #print(labels_frame2class.shape)
            labels_frame2class_all = torch.cat((labels_frame2class_all, labels_frame2class), dim=0)
            pred_clip_all = torch.cat((pred_clip_all, pred_clip), dim=0)

    # Remove the hook
    hook_handle.remove()
    scaler = MinMaxScaler()
    
    threshold = 0.4
    pred_clip_binary = (pred_clip_all > threshold).int()
    df_pred = pd.DataFrame(pred_clip_binary, columns=[config.id2classes[i] for i in range(pred_clip_binary.shape[1])])
    #df_label = pd.DataFrame(labels_frame2class_all, columns=[config.id2classes[i] for i in range(labels_frame2class_all.shape[1])])
    #tsne_dfs = {}
    for i, perplexity in enumerate(perplexities):
        for j, iteration in enumerate(iterations):
            if args.model == "panns":
                X_embedded = TSNE(n_components=2, perplexity=perplexity, n_iter=iteration).fit_transform(embeddings["panns"].view(len_dataset, -1))
                
            elif args.model == "hts-at" or args.model == "htsat":
                X_embedded = TSNE(n_components=2, perplexity=perplexity, n_iter=iteration).fit_transform(embeddings["htsat"].view(len_dataset, -1))
                #print(X_embedded)
            

            data = pd.DataFrame(X_embedded, columns = ["X", "Y"])
            #scaler.fit(X_embedded)
            #X_embedded_scaled = scaler.transform(X_embedded)
            #data_scaled = pd.DataFrame(X_embedded_scaled, columns = ["X", "Y"])
            result_df = pd.concat([data, df_pred], axis=1)
            #result_df_scaled = pd.concat([data_scaled, df_label], axis=1)
            filename_csv = "tsne_" + str(perplexity) + "_" + str(iteration) + ".csv"
            #filename_csv_scaled = "tsne_scaled_" + str(perplexity) + "_" + str(iteration)
            print("Saving file: " +(filename_csv))
            result_df.to_csv(os.path.join(tsne_dir, filename_csv), sep = "\t")
            #result_df_scaled.to_csv(os.path.join(tsne_dir, filename_csv_scaled), sep = "\t")
            #tsne_dfs[perplexity] = {iteration: result_df}
            data = pd.DataFrame
            result_df = pd.DataFrame


    ##SAve as csv for each iteration and perplexity

    ## Make new python file to load the csv and plot the embeddings for each class
    # Create a scatter plot for each label where the label is equal to 1

    """label_columns = [col for col in result_df.columns if col in config.classes2id.keys()]

    label_colors = sns.color_palette('husl', len(label_columns))

    plt.figure(figsize=(10, 8))
    for i, label_column in enumerate(label_columns):
    # Filter the DataFrame to include only rows where the label column is 1.0
        filtered_data = data_expanded[data_expanded[label_column] == 1.0]
    
    # Check if there are any points to plot
        if not filtered_data.empty:
            sns.scatterplot(
            x='X', y='Y',
            data=filtered_data,
            alpha=0.7,
            label=label_column,
            color=label_colors[i]
        )

    plt.title('Scatter Plot for Data Points with Label = 1.0')
    plt.legend()
    plt.show()
    plt.savefig('path/to/save/image.png')"""
