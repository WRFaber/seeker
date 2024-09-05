import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from agent_network import AgentNetwork
from controlled_environment import generate_episode
from helpers import sma, validate_episodes
from sensor import Sensor

# Folder Path
path = os.getenv("DATA_PATH")

# Change the directory
os.chdir(path)

# GENERAL INITIALIZATION
ACTIONS = ["up", "down", "left", "right"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPISODES = 10000
STEPS_TRIGGER = 1000
LOAD_MODEL = False
BATCH = 1

# POLICY INITIALIZATION
dim_hidden_layers = 128
num_hidden_layers = 12
input_dimension = 3
output_dimension = 25
agent_network = AgentNetwork(
    input_dimension, dim_hidden_layers, output_dimension, num_hidden_layers
)
agent_network.to(DEVICE)
agent_network.float()
lengths = []
gamma = 0.99
lr_agent_network = 2**-16
optimizer = torch.optim.Adam(agent_network.parameters(), lr=lr_agent_network)
total_rewards = []
total_observable = []
total_capture_ratio = []

# SENSOR INITIALIZATION
n = 90  # Elevation Limit
m = 360  # Azimuth Limit
trigger = 0

# DATA INITIALIZATION
sample_paths = []
# iterate through all file
for file in os.listdir():
    # Check file is in json format
    if file.endswith(".json"):
        file_path = f"{path}/{file}"
        sample_paths.append(file_path)
training_sample_paths = sample_paths[:350]
valid_paths = validate_episodes(training_sample_paths)

# AGENT LOAD (IF TRAINING FROM NON-INITIAL STARTING POINT)
path = os.getenv("MODEL_PATH")
os.chdir(path)
if LOAD_MODEL:
    agent_network = torch.load("./Seeker_Network.pth")
    agent_network.to(DEVICE)


# MAIN LOOP
for episode_num in tqdm(range(EPISODES)):
    trigger += 1
    sensor = Sensor()
    agent_network.saved_log_probs = []

    # Sample batch according to current policy
    policy_sample_episodes = [
        list(
            generate_episode(
                sensor,
                training_sample_paths,
                agent_network=agent_network,
                device=DEVICE,
                horizon=20,
                max_episode_len=1000,
            )
        )
        for _ in range(BATCH)
    ]

    # Update policy and record performance metrics
    returns = []
    policy_loss = []
    for episode in policy_sample_episodes:
        # Record episode performance
        lengths.append(len(episode))
        episode_rewards = [step[0][2] for step in episode]
        total_rewards.append(sum(episode_rewards))
        observable_objects = []
        [observable_objects.append(x) for step in episode for x in step[2]]
        observable_objects = set(observable_objects)
        detected_objects = []
        [detected_objects.append(x) for step in episode for x in step[3]]
        detected_objects = set(detected_objects)
        total_observable = len(observable_objects)
        total_detected = len(detected_objects)
        total_capture_ratio.append(total_detected / total_observable)
        # Update policy
        for t, (
            (state, action, reward),
            log_probs,
            observable_ids,
            detected_ids,
            looks,
        ) in enumerate(episode):
            gammas_vec = gamma ** (torch.arange(t, len(episode)) - t)
            # Rewards are different for each step depending on how well the tasking did
            G = torch.sum(
                gammas_vec * torch.asarray(episode_rewards[-len(gammas_vec) :])
            )
            returns.append(G.item())
            policy_loss.append(log_probs)
    if policy_loss:
        returns = (returns - np.mean(returns)) / (np.std(returns) + 10e-3)
        update_param = []
        for log_prob, R in zip(policy_loss, returns):
            update_param.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(update_param).sum()
        policy_loss.backward()
        optimizer.step()

    # Save agent and plot performance
    if trigger == STEPS_TRIGGER:
        trigger = 0
        torch.save(agent_network, "./Seeker_Network.pth")
        ave = sma(total_capture_ratio, lag=100)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle("Agent Performance")
        ax1.plot(range(len(total_capture_ratio)), total_capture_ratio, "o-")
        ax1.set_ylabel("Capture Ratio")
        ax1.plot(range(len(ave)), ave)

        ax2.plot(range(len(total_rewards)), total_rewards, ".-")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Total Reward")
        plt.savefig(f"Performance_Stacked_N{episode_num +1}.png")
        # plt.show()
        # plt.plot(look_directions)


stop = 0
