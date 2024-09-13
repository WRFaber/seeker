import os
from math import cos, sin

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from controlled_environment import Seeker_Environment
from helpers import validate_episodes
from ppo import PPO
from sensor import Sensor

####### initialize environment hyperparameters ######
path = os.getenv("DATA_PATH")                              # Read data path from .env

os.chdir(path)                                             # Change path to data path

has_continuous_action_space = False                        # continuous action space; else discrete

max_ep_len = 1000                                          # max timesteps in one episode
max_training_timesteps = int(3e6)                          # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10                               # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2                                  # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)                                 # save model frequency (in num timesteps)

action_std = 0.6                                           # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05                               # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                                       # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)                         # action_std decay frequency (in num timesteps)

sample_paths = []
for file in os.listdir():                                  # iterate through all files in data path
    if file.endswith(".json"):                             # Check file is in json format
        file_path = f"{path}/{file}"                       #
        sample_paths.append(file_path)                     # If json add to sample paths
training_sample_paths = sample_paths[:350]                 # use first 350 as training paths remaing will be used for test
valid_paths = validate_episodes(training_sample_paths)     # ensure training paths are valid fragmentations that are visible from sensor

#####################################################

## Note : print/log frequencies should be > than max_ep_len

################ PPO hyperparameters ################
update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)
#####################################################
env_name = 'seeker'

os.chdir("./")
sensor = Sensor()
env = Seeker_Environment(valid_paths,sensor)

# state space dimension
state_dim = 3

# action space dimension
if has_continuous_action_space:
    # action_dim = env.action_space.shape[0]
    ...
else:
    action_dim = 25

load_model = True
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

path = os.getenv("MODEL_PATH")
directory = path + '/' + env_name + '/'
random_seed = 0
run_num_pretrained = 0
checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
if load_model:
    ppo_agent.load(checkpoint_path)


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
            az_rad = np.deg2rad(walk[i][0][0].to("cpu")[0].item())
            el_rad = np.deg2rad(walk[i][0][0].to("cpu")[1].item())
            z = sin(el_rad)
            y = sin(az_rad) * cos(el_rad)
            x = cos(az_rad) * cos(el_rad)
            steps.append([x, y, z])
        line.set_data_3d(np.array(steps).T)
        looks = []
        if walk[num][1] != None:
            for look in walk[num][1][1]:
                az_rad = np.deg2rad(look['azimuth'])
                el_rad = np.deg2rad(look['elevation'])
                z = sin(el_rad)
                y = sin(az_rad) * cos(el_rad)
                x = cos(az_rad) * cos(el_rad)
                looks.append([x, y, z])
        else:
            looks = []
        if len(looks) == 0:
            ...
        else:
            x_data = np.array(looks).T[0, :]
            y_data = np.array(looks).T[1, :]
            z_data = np.array(looks).T[2, :]
            dot._offsets3d = (x_data, y_data, z_data)
    return lines, dots


# Data: 40 random walks as (num_steps, 3) arrays
def generate_episode():
    state = env.reset()
    done = False
    episode = []
    while not done:
        action = ppo_agent.select_action(state)
        state, _, done, looks = env.step(action)
        if not done:
            episode.append((state,looks))
    return episode

walks = [list(generate_episode()) for index in range(10)]

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

path = os.getenv("GIF_PATH")
ani.save(filename=path+ "/" + env_name + "/trained.gif", writer="pillow")