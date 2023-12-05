import pandas as pd
import numpy as np
import torch
from panns_models import *
from model.htsat import HTSAT_Swin_Transformer
import warnings
import os
import torchaudio
import logging


def get_model(model_name, config: dict, weights_path: str):
    if model_name == "hts-at" or args.model == "htsat":
        model = HTSAT_Swin_Transformer(
        spec_size=configs["hts-at"]["htsat_spec_size"],
        patch_size=configs["hts-at"]["htsat_patch_size"],
        in_chans=1,
        num_classes=configs["classes_num"],
        window_size=configs["hts-at"]["htsat_window_size"],
        config = config,
        depths = configs["hts-at"]["htsat_depth"],
        embed_dim = configs["hts-at"]["htsat_dim"],
        patch_stride=tuple(configs["hts-at"]["htsat_stride"]),
        num_heads=configs["hts-at"]["htsat_num_head"]
    )
        
    elif model_name == "panns":
        # model
        model_config = {
            "sample_rate": 16000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 10
                }
        model = PANNsCNN14Att(**model_config)
        #weights = torch.load("Cnn14_DecisionLevelAtt_mAP0.425.pth", map_location = "cpu")
        # Fixed in V3
        #model.load_state_dict(weights["model"])
        model.att_block = AttBlock(2048, 10, activation='sigmoid')
        
    
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.basicConfig(level=logging.INFO) 
    logging.info("device: " +device)
    model.to(device)
    model.eval()
    return model

def prediction_for_clip(SR,
               max_audio_DURATION,
               model_name,
               test_df: pd.DataFrame,
                clip: np.ndarray, 
                model,
                threshold=0.5):
    PERIOD = max_audio_DURATION
    audios = []
    y = clip.astype(np.float32)
    len_y = len(y)
    start = 0
    end = PERIOD * SR
    while True:
        y_batch = y[start:end].astype(np.float32)
        if len(y_batch) != PERIOD * SR:
            y_pad = np.zeros(PERIOD * SR, dtype=np.float32)
            y_pad[:len(y_batch)] = y_batch
            audios.append(y_pad)
            break
        start = end
        end += PERIOD * SR
        audios.append(y_batch)
        
    array = np.asarray(audios)
    tensors = torch.from_numpy(array)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    estimated_event_list = []
    global_time = 0.0
    site = test_df["site"].values[0]
    audio_id = test_df["audio_id"].values[0]
    for image in progress_bar(tensors):
        image = image.view(1, image.size(0))
        image = image.to(device)

        with torch.no_grad():
            prediction = model(image)
            framewise_outputs = prediction["framewise_output"].detach(
                ).cpu().numpy()[0]
                
        thresholded = framewise_outputs >= threshold

        for target_idx in range(thresholded.shape[1]):
            if thresholded[:, target_idx].mean() == 0:
                pass
            else:
                detected = np.argwhere(thresholded[:, target_idx]).reshape(-1)
                head_idx = 0
                tail_idx = 0
                while True:
                    if (tail_idx + 1 == len(detected)) or (
                            detected[tail_idx + 1] - 
                            detected[tail_idx] != 1):
                        onset = 0.01 * detected[
                            head_idx] + global_time
                        offset = 0.01 * detected[
                            tail_idx] + global_time
                        onset_idx = detected[head_idx]
                        offset_idx = detected[tail_idx]
                        max_confidence = framewise_outputs[
                            onset_idx:offset_idx, target_idx].max()
                        mean_confidence = framewise_outputs[
                            onset_idx:offset_idx, target_idx].mean()
                        estimated_event = {
                            "site": site,
                            "audio_id": audio_id,
                            "ebird_code": INV_BIRD_CODE[target_idx],
                            "onset": onset,
                            "offset": offset,
                            "max_confidence": max_confidence,
                            "mean_confidence": mean_confidence
                        }
                        estimated_event_list.append(estimated_event)
                        head_idx = tail_idx + 1
                        tail_idx = tail_idx + 1
                        if head_idx >= len(detected):
                            break
                    else:
                        tail_idx += 1
        global_time += PERIOD
        
    prediction_df = pd.DataFrame(estimated_event_list)
    return prediction_df

def prediction(SR,
               max_audio_DURATION,
               model_name,
               test_df: pd.DataFrame,
               test_audio_path: Path,
               model_config: dict,
               weights_path: str,
               threshold=0.5):
    model = get_model(model_name, model_config, weights_path)
    unique_audio_id = test_df.filename.unique()

    warnings.filterwarnings("ignore")
    prediction_dfs = []
    for audio_id in unique_audio_id:
        print(f"Loading {audio_id}")
        clip, _ = torchaudio.load(os.path.join(test_audio_path, audio_id))
        
        test_df_for_audio_id = test_df.query(
            f"filename == '{audio_id}'").reset_index(drop=True)
        print(f"Prediction on {audio_id}")
        prediction_df = prediction_for_clip(test_df_for_audio_id,
                                                clip=clip,
                                                model=model,
                                                threshold=threshold)

        prediction_dfs.append(prediction_df)
    
    prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)
    return prediction_df

