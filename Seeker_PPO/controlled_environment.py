import random
from datetime import timedelta

import torch
from helpers import (
    format_fragment_json_degrees,
    read_text_file,
)
from sensor import Sensor


def generate_episode(
    sensor: Sensor,
    sample_paths,
    agent_network: AgentNetwork,
    device="cpu",
    horizon=10,
    max_episode_len=100,
    record_looks=False,
):
    sample = random.sample(sample_paths, 1)
    raw_data = read_text_file(sample[0])
    fragmentation = format_fragment_json_degrees(raw_data)
    offset = 0
    reward = 0
    initial_state = (
        round(fragmentation[0][1][0]["azimuth"]) + offset,
        round(fragmentation[0][1][0]["elevation"]) + offset,
    )
    sensor.initialize_look_direction(initial_state)
    state = sensor.get_state(device, reward)
    ep_length = 0
    max_date = fragmentation[-1][0]
    current_date = fragmentation[0][0]
    objects_detected = []
    rolling_horizon = None
    while current_date < max_date:
        # Convert state to tensor and pass through policy network to get action probabilities
        ep_length += 1
        if rolling_horizon is None:
            rolling_horizon = state
        else:
            rolling_horizon = torch.vstack((rolling_horizon, state))
        if len(rolling_horizon) > horizon:
            rolling_horizon = rolling_horizon[-horizon:]
        else:
            ...
        action = agent_network.select_action(rolling_horizon)

        # Take the action and get the new state and reward
        sensor.move(action)
        available_looks = None
        for x in fragmentation:
            if x[0] == current_date:
                available_looks = x
        objects_observable = []
        new_objects_detected = []
        looks = []
        if available_looks is not None:
            for x in available_looks[1]:
                objects_observable.append(x["objectId"])
                az_diff = abs(sensor.look_direction[0] - x["azimuth"])
                el_diff = abs(sensor.look_direction[1] - x["elevation"])
                if az_diff < sensor.az_fov and el_diff < sensor.el_fov:
                    if len(objects_detected) == 0:
                        reward += 1.0
                        objects_detected.append(x["objectId"])
                        new_objects_detected.append(x["objectId"])
                    else:
                        new_object = objects_detected.count(x["objectId"])
                        if new_object == 0:
                            reward += 1.0
                            objects_detected.append(x["objectId"])
                            new_objects_detected.append(x["objectId"])
                        else:
                            reward += 0.25
                else:
                    if len(objects_detected) == 0:
                        reward -= 0.1
                    else:
                        eluded = objects_detected.count(x["objectId"])
                        if eluded == 0:
                            reward -= 0.1
                if record_looks:
                    look = [x["azimuth"], x["elevation"]]
                    looks.append(look)
        else:
            reward = 0
        # Add the state, action, and reward to the episode
        new_episode_sample = (state, action, reward)
        yield (
            new_episode_sample,
            agent_network.saved_log_probs[-1],
            list(set(objects_observable)),
            new_objects_detected,
            looks,
        )

        # Update the current state
        state = sensor.get_state(device, reward)
        current_date = current_date + timedelta(seconds=10)
