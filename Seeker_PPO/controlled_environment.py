import random
from datetime import timedelta

from helpers import (
    format_fragment_json_degrees,
    read_text_file,
)
from sensor import Sensor


class Seeker_Environment:
    def __init__(self, paths, sensor: Sensor, device="cpu",):
        self.sample_paths = paths
        self.device = device
        self.sample = random.sample(paths)
        self.raw_data = read_text_file(self.sample[0])
        self.fragmentation = format_fragment_json_degrees(self.raw_data)
        self.offset = 0
        self.reward = 0
        self.initial_state = (
            round(self.fragmentation[0][1][0]["azimuth"]) + self.offset,
            round(self.fragmentation[0][1][0]["elevation"]) + self.offset,
        )
        self.current_date = self.fragmentation[0][0]
        self.max_date = self.fragmentation[-1][0]
        self.sensor = sensor
        self.sensor.initialize_look_direction(self.initial_state)
        self.ep_length = 0
        self.objects_detected = []

    def reset(self):
        self.sample = random.sample(self.sample_paths,1)
        self.raw_data = read_text_file(self.sample[0])
        self.fragmentation = format_fragment_json_degrees(self.raw_data)
        self.offset = 0
        self.reward = 0
        self.initial_state = (
            round(self.fragmentation[0][1][0]["azimuth"]) + self.offset,
            round(self.fragmentation[0][1][0]["elevation"]) + self.offset,
        )
        self.current_date = self.fragmentation[0][0]
        self.max_date = self.fragmentation[-1][0]
        self.sensor.initialize_look_direction(self.initial_state)
        self.ep_length = 0
        self.objects_detected = []
        
    def step(self, action):
        self.reward = 0.0
        self.ep_length += 1
        if self.current_date < self.max_date:
            done = False
            # Take the action and get the new state and reward
            self.sensor.move(action)
            available_looks = None
            for x in self.fragmentation:
                if x[0] == self.current_date:
                    available_looks = x
            self.objects_observable = []
            self.new_objects_detected = []
            if available_looks is not None:
                for x in available_looks[1]:
                    self.objects_observable.append(x["objectId"])
                    az_diff = abs(self.sensor.look_direction[0] - x["azimuth"])
                    el_diff = abs(self.sensor.look_direction[1] - x["elevation"])
                    if az_diff < self.sensor.az_fov and el_diff < self.sensor.el_fov:
                        if len(self.objects_detected) == 0:
                            self.reward += 1.0
                            self.objects_detected.append(x["objectId"])
                            self.new_objects_detected.append(x["objectId"])
                        else:
                            new_object = self.objects_detected.count(x["objectId"])
                            if new_object == 0:
                                self.reward += 1.0
                                self.objects_detected.append(x["objectId"])
                                self.new_objects_detected.append(x["objectId"])
                            else:
                                self.reward += 0.25
                    else:
                        if len(self.objects_detected) == 0:
                            self.reward -= 0.1
                        else:
                            eluded = self.objects_detected.count(x["objectId"])
                            if eluded == 0:
                                self.reward -= 0.1
            else:
                self.reward = 0
            # Update the current state
            self.state = self.sensor.get_state(self.device, self.reward)
            self.current_date = self.current_date + timedelta(seconds=10)
        else: 
            done = True
        yield (
                self.state,
                self.reward,
                done
            )