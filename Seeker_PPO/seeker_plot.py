import os
from math import cos, sin

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from controlled_environment import generate_episode
from sensor import Sensor

# Folder Path
path = os.getenv("DATA_PATH")

# Change the directory
os.chdir(path)

# INITIALIZATION
ACTIONS = ["up", "down", "left", "right"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPISODES = 5000
# POLICY INITIALIZATION
load_model = True
dim_hidden_layers = 128
num_hidden_layers = 2
input_dimension = 3
output_dimension = 25
agent_network = AgentNetwork(
    input_dimension, dim_hidden_layers, output_dimension, num_hidden_layers
)
agent_network.to(DEVICE)
lengths = []
rewards = []
gamma = 0.99
lr_policy_net = 2**-16
optimizer = torch.optim.Adam(agent_network.parameters(), lr=lr_policy_net)


# SENSOR PARAMETERS
sensor = Sensor()

sample_paths = []
# iterate through all file
for file in os.listdir():
    # Check file is in json format
    if file.endswith(".json"):
        file_path = f"{path}/{file}"
        sample_paths.append(file_path)
sample_paths = sample_paths

# Fixing random state for reproducibility
np.random.seed(19680801)

path = os.getenv("MODEL_PATH")
os.chdir(path+"/Seeker_PPO")
if load_model:
    agent_network = torch.load("./Seeker_Network.pth")
    agent_network.to(DEVICE)


def random_walk(num_steps, max_step=0.05):
    """Return a 3D random walk as (num_steps, 3) array."""
    start_pos = np.random.random(3)
    steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
    walk = start_pos + np.cumsum(steps, axis=0)
    return walk


def update_lines(num, walks, lines):
    for line, dot, walk in zip(lines, dots, walks):
        steps = []
        for i in range(num + 1):
            az_rad = np.deg2rad(walk[i][0][0].to("cpu").numpy()[0][0])
            el_rad = np.deg2rad(walk[i][0][0].to("cpu").numpy()[0][1])
            z = sin(el_rad)
            y = sin(az_rad) * cos(el_rad)
            x = cos(az_rad) * cos(el_rad)
            steps.append([x, y, z])
        line.set_data_3d(np.array(steps).T)
        looks = []
        for look in walk[num][4]:
            az_rad = np.deg2rad(look[0])
            el_rad = np.deg2rad(look[1])
            z = sin(el_rad)
            y = sin(az_rad) * cos(el_rad)
            x = cos(az_rad) * cos(el_rad)
            looks.append([x, y, z])
        if len(looks) == 0:
            ...
        else:
            x_data = np.array(looks).T[0, :]
            y_data = np.array(looks).T[1, :]
            z_data = np.array(looks).T[2, :]
            dot._offsets3d = (x_data, y_data, z_data)
    return lines, dots


# Data: 40 random walks as (num_steps, 3) arrays

walks = [
    list(
        generate_episode(
            sensor,
            sample_paths,
            agent_network=agent_network,
            device=DEVICE,
            max_episode_len=1000,
            record_looks=True,
        )
    )
    for index in range(10)
]

# longer_walks = []
# for walk in walks:
#     if len(walk) < 50 and len(walk) > 20:
#         longer_walks.append(walk)
# walks = longer_walks
num_steps = min([len(walk) for walk in walks])

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Create lines initially without data
lines = [ax.plot([], [], [])[0] for _ in walks]
dots = [ax.scatter([], [], []) for _ in walks]

# Setting the Axes properties
ax.set(xlim3d=(-1, 1), xlabel="X")
ax.set(ylim3d=(-1, 1), ylabel="Y")
ax.set(zlim3d=(0, 1), zlabel="Z")

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(walks, lines), interval=100
)

plt.show()
