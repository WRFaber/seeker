import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import (
    format_fragment_json_degrees,
    read_text_file,
)


def analyze_fragmentations(
    sample_paths,
):
    episode_duration = []
    opportunities_per_duration = []
    ave_len_available_looks = []
    id = 1
    ids = []
    for sample in sample_paths:
        ids.append(id)
        raw_data = read_text_file(sample)
        fragmentation = format_fragment_json_degrees(raw_data)
        max_date = fragmentation[-1][0]
        current_date = fragmentation[0][0]
        delta = max_date - current_date
        episode_duration.append(delta.total_seconds())
        opportunities_per_duration.append(len(fragmentation) / episode_duration[-1])
        len_available_looks = []
        for x in fragmentation:
            available_looks = x[1]
            len_available_looks.append(len(available_looks))
        id += 1
        ave_len_available_looks.append(np.average(len_available_looks))

    t = range(0, len(sample_paths))
    data = list(
        zip(t, episode_duration, opportunities_per_duration, ave_len_available_looks)
    )
    df = pd.DataFrame(
        data,
        columns=[
            "Episode",
            "Episode Duration",
            "Number of Opportunities",
            "Average Available Looks",
        ],
    )
    return df


# DATA INITIALIZATION
# Folder Path
path = os.getenv("DATA_PATH")

# Change the directory
os.chdir(path)
sample_paths = []
# iterate through all file
for file in os.listdir():
    # Check file is in json format
    if file.endswith(".json"):
        file_path = f"{path}/{file}"
        sample_paths.append(file_path)
training_sample_paths = sample_paths[:350]
t = range(0, len(training_sample_paths))

df = analyze_fragmentations(training_sample_paths)
df = df[df["Episode Duration"] > 600]

t = range(0, len(df))
df = df.sort_values("Episode Duration")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.suptitle("Fragmentations Analysis")
fig.suptitle("Episode Durations")
ax1.plot(t, df["Episode Duration"])
ax1.set_ylabel("Duration (S)")

ax2.plot(t, df["Number of Opportunities"], ".-")
ax2.set_ylabel("Opportunities in FOR")

ax3.plot(t, df["Average Available Looks"])
ax3.set_xlabel("Episode")
ax3.set_ylabel("Objects Per Opportunity")
plt.show()
stop = 1
