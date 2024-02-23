import numpy as np
import os
import pandas as pd
from pathlib import Path
import config
import yaml
import re

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir_path", type=str, help="input directory path", default="./"
    )
    args = parser.parse_args()
    print(args.dir_path)
    output_dir = os.path.join(args.dir_path, "umap_plots")
    pattern = r"\w\_(\d+)\_(\d+\.\d+)"
    filenames = os.listdir(args.dir_path)
    print(filenames)
    print("****************")
    print(output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for file_name in filenames:
        filepath = os.path.join(args.dir_path, file_name)
        if os.path.isfile(filepath):
            #print(filepath)
            match = re.search(pattern, file_name)
            if match:
                neighbour = match.group(1)
                distance = match.group(2)

            #print("perplexity = " +str(per))
            #print("iteration = " +str(ite))        
            df = pd.read_csv(filepath, sep = "\t")
            df = df.drop('Unnamed: 0', axis=1)
            label_columns = [col for col in df.columns if col in config.classes2id.keys()]

            label_colors = sns.color_palette('tab10', n_colors=len(label_columns))

            plt.figure(figsize=(10, 8))
            for i, label_column in enumerate(label_columns):
            # Filter the DataFrame to include only rows where the label column is 1.0
                filtered_data = df[df[label_column] == 1.0]
                if not filtered_data.empty:
                    sns.scatterplot(
                    x='X', y='Y',
                    data=filtered_data,
                    alpha=0.7,
                    label=label_column,
                    color=label_colors[i]
                    )

            plt.title('Scatter Plot for UMAP Neighbour: ' +str(neighbour) + " and distance:" + str(distance))
            plt.legend()
            plt.show()
            plt.savefig(os.path.join(output_dir, str(neighbour)+ "_"+ str(distance) +'.png'))
            print("saved: " +os.path.join(output_dir, str(neighbour)+ "_"+ str(distance) +'.png'))
            plt.close()
