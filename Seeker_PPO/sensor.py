import torch


class Sensor:
    def __init__(
        self,
        az_max=360,
        el_max=90,
        az_fov=2,
        el_fov=2,
        pan_rate_el=1,
        pan_rate_az=3,
        look_direction=None,
    ):
        self.n = el_max  # Max elevation in degrees assuming min is 0 (i.e. traditional 0-n degrees elevation )
        self.m = az_max  # Max azimuth in degrees assuming min is 0 ( i.e. traditional 0-m degrees azimuthm)
        self.el_fov = el_fov
        self.az_fov = az_fov
        self.look_direction = look_direction
        self.pan_rate_az = pan_rate_az
        self.pan_rate_el = pan_rate_el
        self.az_map = [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5]
        self.el_map = [1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5]
        self.directions_az = []
        self.directions_el = []
        for x in range(3):
            for y in range(len(self.el_map)):
                self.directions_az.append(x * self.pan_rate_az * self.az_map[y])
                self.directions_el.append(x * self.pan_rate_el * self.el_map[y])
        self.directions_az.append(0)
        self.directions_el.append(0)

    # def move(self, direction):
    #     x, y = self.look_direction
    #     if direction == "up":
    #         if y < self.n - self.pan_rate:
    #             self.look_direction = (x, y + self.pan_rate)
    #     elif direction == "down":
    #         if y >= self.pan_rate:
    #             self.look_direction = (x, y - self.pan_rate)
    #     elif direction == "left":
    #         if x > 0:
    #             x -= self.pan_rate
    #             if x < 0:  # Accounts for circular horizontal movement
    #                 x += self.m
    #             self.look_direction = (x, y)
    #     elif direction == "right":
    #         if x <= self.m:
    #             x += self.pan_rate
    #             if x > self.m:  # Accounts for circular horizontal movement
    #                 x -= self.m
    #             self.look_direction = (x, y)

    def move(self, action):
        x, y = self.look_direction
        # d = ceil(action / len(self.directions_az))
        # r = action % len(self.directions_az)
        direction_az = self.directions_az[action]
        direction_el = self.directions_el[action]
        y = y + (direction_el)
        x = x + (direction_az)
        if y > self.n:
            y = self.n
        elif y < 0:
            y = 0
        if x > self.m:
            x = x - self.m
        elif x < 0:
            x = x + self.m
        self.look_direction = (x, y)

    def get_state(self, device, reward):
        return (
            torch.FloatTensor((self.look_direction[0], self.look_direction[1], reward))
            .unsqueeze(0)
            .to(device)
        )

    def initialize_look_direction(self, direction):
        self.look_direction = direction
