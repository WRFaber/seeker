import json
from datetime import datetime

import numpy as np
import torch
from torch.autograd import grad


def gradients_wrt_params(net: torch.nn.Module, loss_tensor: torch.Tensor):
    # Dictionary to store gradients for each parameter
    # Compute gradients with respect to each parameter
    for name, param in net.named_parameters():
        g = grad(loss_tensor, param, retain_graph=True)[0]
        param.grad = g


def update_params(net: torch.nn.Module, lr: float) -> None:
    # Update parameters for the network
    for name, param in net.named_parameters():
        param.data += lr * param.grad


def read_text_file(file_path):
    # This method helps load json files
    f = open(file_path)
    return json.load(f)


def format_fragment_json(file):
    # Formats raw data into usable information
    vis_windows = file["timeBasedVisWindows"]
    frag_windows = [
        (datetime.strptime(x["date"], "%Y-%m-%dT%H:%M:%S.%f"), x["availableLooks"])
        for x in vis_windows
    ]
    frag_windows.sort()
    return frag_windows


def format_fragment_json_degrees(file):
    # Formats raw data into usable information
    vis_windows = file["timeBasedVisWindows"]
    frag_windows = [
        (
            datetime.strptime(x["date"], "%Y-%m-%dT%H:%M:%S.%f"),
            [
                {
                    "objectId": y["objectId"],
                    "azimuth": np.rad2deg(y["azimuth"]),
                    "elevation": np.rad2deg(y["elevation"]),
                }
                for y in x["availableLooks"]
            ],
        )
        for x in vis_windows
    ]
    frag_windows.sort()
    return frag_windows


def sma(data, lag=3):
    sma = []
    for i in range(lag):
        sma.append(np.nan)
    for i in range(lag, len(data)):
        sma.append(np.mean(data[i - lag : i]))
    return np.array(sma)


def validate_episodes(sample_paths, looks_threshold=15, duration_threshold=600):
    valid_paths = []
    for sample in sample_paths:
        raw_data = read_text_file(sample)
        fragmentation = format_fragment_json_degrees(raw_data)
        max_date = fragmentation[-1][0]
        current_date = fragmentation[0][0]
        delta = max_date - current_date
        episode_duration = delta.total_seconds()
        len_available_looks = len(fragmentation)
        if (
            len_available_looks > looks_threshold
            and episode_duration > duration_threshold
        ):
            valid_paths.append(sample)

    return valid_paths
