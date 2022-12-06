import pygame
from pygame.sprite import Sprite, AbstractGroup
from pygame_ped_env.spritesheet import SpriteSheet

from pygame_ped_env.customsprites import Vehicle, Pedestrian

import numpy as np


class RLVehicle(Vehicle):
    # class for hooking in RL algorithms
    def __init__(
        self, window_size, vehicleClass: str, direction: str, *groups: AbstractGroup
    ) -> None:

        if direction == "right":
            start = [0, window_size[1] / 2]
            end = [window_size[0], window_size[1] / 2]
        elif direction == "left":
            start = [window_size[0], window_size[1] / 2]
            end = [0, window_size[1] / 2]
        else:
            raise AttributeError(
                'Only "left" and "right" directions are implemented for vehicles'
            )

        super().__init__(start[0], start[1], 2, vehicleClass, direction, *groups)

        start = self.rect.center
        # ensure goal is centred in lane after car sprite position is adjusted in super
        end[1] = start[1]

        # hook for RL algo
        self.model = None

        self.moved = [np.array([0, 0])]
        self.init_pos = np.array(start)
        self.objective = np.array(end)

        self.action_map = {
            0: np.array([0, 1]),  # "down"
            1: np.array([-1, 1]),  # "downleft"
            2: np.array([-1, 0]),  # "left"
            3: np.array([-1, -1]),  # "upleft"
            4: np.array([0, -1]),  # "up"
            5: np.array([1, -1]),  # "upright"
            6: np.array([1, 0]),  # "right"
            7: np.array([1, 1]),  # "downright"
        }
        self.speed_map = {
            0: np.floor(self.speed * (1 / 3)),  # "slow"
            1: np.floor(self.speed * (2 / 3)),  # "medium"
            2: np.floor(self.speed * (1)),  # "fast"
            3: 0.0,  # "nomove"
        }

    def dist_to_objective(self, pos=np.array([None, None])):
        if pos.any():
            return np.abs(pos - self.objective)
        else:
            return np.abs(self.rect.center - self.objective)

    def act(self, action):
        mapped_action = (
            self.action_map[action[0][0] % 8]
            * self.speed_map[np.floor(action[0][0] / 8)]
        )
        self.rect.center += mapped_action
        self.moved.append(mapped_action)

    def update(self, action=None):  # for non learning agent
        # self.rect.center += np.array([-1, 0]) # just move left
        if action:
            mapped_action = (
                self.action_map[action[0][0] % 8]
                * self.speed_map[np.floor(action[0][0] / 8)]
            )
            self.rect.center += mapped_action
            self.moved.append(mapped_action)
        else:
            pass

    @classmethod
    def init_from_scenario(cls, scenario: str, window_size):
        if scenario == ("RL_left" or "SRL_left"):
            return cls(window_size, "car", "left")
        else:
            return cls(window_size, "car", "right")


class RandomVehicle(Vehicle):
    def __init__(
        self, window_size, vehicleClass: str, direction: str, *groups: AbstractGroup
    ) -> None:
        if direction == "right":
            start = [0, window_size[1] / 2]
        elif direction == "left":
            start = [window_size[0], window_size[1] / 2]
        else:
            raise AttributeError(
                'Only "left" and "right" directions are implemented for vehicles'
            )

        super().__init__(start[0], start[1], 2, vehicleClass, direction, *groups)

    def update(self):

        direction = self.get_direction()

        def polarRandom(dx, dy):

            epsilon = ((np.random.rand() * 2) - 1) / 200
            # epsilon = 0

            r = np.sqrt(((self.rect.x + dx) ** 2 + (self.rect.y + dy) ** 2))
            theta = np.arctan((self.rect.y + dy) / (self.rect.x + dx))

            self.rect.x = np.round(r * np.cos(theta + epsilon))
            self.rect.y = np.round(r * np.sin(theta + epsilon))

        if direction == "up":
            polarRandom(0, -self.speed)
        elif direction == "down":
            polarRandom(0, self.speed)
        elif direction == "left":
            polarRandom(-self.speed, 0)
        elif direction == "right":
            polarRandom(self.speed, 0)

        self.animate(direction)

    def act(self, action):
        self.update()


class KeyboardVehicle(Vehicle):
    def __init__(
        self, window_size, vehicleClass: str, direction: str, *groups: AbstractGroup
    ) -> None:
        if direction == "right":
            start = [0, window_size[1] / 2]
        elif direction == "left":
            start = [window_size[0], window_size[1] / 2]
        else:
            raise AttributeError(
                'Only "left" and "right" directions are implemented for vehicles'
            )

        super().__init__(start[0], start[1], 2, vehicleClass, direction, *groups)

    def update(self):
        assert pygame.get_init(), "Keyboard agent cannot be used while headless"

        keys = pygame.key.get_pressed()

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.rect.y += self.speed
            self.animate("down")
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.rect.y -= self.speed
            self.animate("up")
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.rect.x += self.speed
            self.animate("right")
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.rect.x -= self.speed
            self.animate("left")

    @classmethod
    def init_from_scenario(cls, scenario: str, window_size):
        if scenario == "H_left":
            return cls(window_size, "car", "left")
        else:
            return cls(window_size, "car", "right")


class RandomPedestrian(Pedestrian):
    def __init__(self, x, y, init_direction, *groups: AbstractGroup) -> None:
        super().__init__(x, y, init_direction, *groups)

    def update(self):

        direction = self.get_direction()

        def polarRandom(dx, dy):
            epsilon = ((np.random.rand() * 2) - 1) / 50
            # epsilon = 0
            if ((self.rect.x + dx) == 0) and ((self.rect.y + dy) == 0):
                self.rect.x += 1
                self.rect.y += 1
            else:
                r = np.sqrt(((self.rect.x + dx) ** 2 + (self.rect.y + dy) ** 2))
                with np.errstate(divide="ignore"):
                    theta = np.arctan(
                        np.divide(np.abs(self.rect.y + dy), np.abs(self.rect.x + dx))
                    )

                if ((self.rect.x + dx) == 0) and ((self.rect.y + dy) == 0):
                    theta = 0
                self.rect.x = np.round(r * np.cos(theta + epsilon))
                self.rect.y = np.round(r * np.sin(theta + epsilon))

        if direction == "up":
            polarRandom(0, -self.speed)
        elif direction == "down":
            polarRandom(0, self.speed)
        elif direction == "left":
            polarRandom(-self.speed, 0)
        elif direction == "right":
            polarRandom(self.speed, 0)

        self.animate(direction)


class KeyboardPedestrian(Pedestrian):
    def __init__(self, x, y, init_direction, *groups: AbstractGroup) -> None:
        super().__init__(x, y, init_direction, *groups)

    def update(self):
        assert pygame.get_init(), "Keyboard agent cannot be used while headless"

        keys = pygame.key.get_pressed()

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.rect.y += self.speed
            self.animate("down")
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.rect.y -= self.speed
            self.animate("up")
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.rect.x += self.speed
            self.animate("right")
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.rect.x -= self.speed
            self.animate("left")
