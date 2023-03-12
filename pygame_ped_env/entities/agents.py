import pygame
from pygame.sprite import Sprite, AbstractGroup
from pygame_ped_env.entities.spritesheet import SpriteSheet

from pygame_ped_env.entities.customsprites import Vehicle, Pedestrian

import numpy as np
import pickle


def genAction_map():
    return {
        0: np.array([0, 1]),  # "down"
        1: np.array([-1, 1]),  # "downleft"
        2: np.array([-1, 0]),  # "left"
        3: np.array([-1, -1]),  # "upleft"
        4: np.array([0, -1]),  # "up"
        5: np.array([1, -1]),  # "upright"
        6: np.array([1, 0]),  # "right"
        7: np.array([1, 1]),  # "downright"
    }


def genSpeed_map(speed):
    return {
        0: np.floor(speed * (1 / 3)),  # "slow"
        1: np.floor(speed * (2 / 3)),  # "medium"
        2: np.floor(speed * (1)),  # "fast"
        3: 0.0,  # "nomove"
    }


class RLVehicle(Vehicle):
    # class for hooking in RL algorithms
    def __init__(
        self,
        window_size,
        vehicleClass: str,
        direction: str,
        *groups: AbstractGroup,
        headless=True,
        colour=None,
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
        colours = [
            "blue",
            "silver",
            "white",
            "red",
            "yellow",
            "green",
            "brown",
            "orange",
        ]
        if colour is None:
            colour = np.random.choice(colours)
        else:
            pass

        super().__init__(
            start[0],
            start[1],
            2,
            vehicleClass,
            direction,
            *groups,
            headless=headless,
            colour=colour,
        )
        if direction == "right":
            self.rect.midleft += np.array(self.rect.size[0])
        elif direction == "left":
            self.rect.midright -= np.array(self.rect.size[0])
        start = self.rect.center
        # ensure goal is centred in lane after car sprite position is adjusted in super
        end[1] = start[1]

        # hook for RL algo
        self.model = None

        self.replay = False
        self.moved = [np.array([0, 0])]
        self.init_pos = np.array(start)
        self.objective = np.array(end)

        self.action_map = genAction_map()
        self.speed_map = genSpeed_map(self.speed)

    def dist_to_objective(self, pos=np.array([None, None])):
        if pos.any():
            return np.abs(pos - self.objective)
        else:
            return np.abs(self.rect.center - self.objective)

    def act(self, action):
        if isinstance(action, np.integer):
            pass
        else:
            action = action[0][0]

        mapped_action = (
            self.action_map[action % 8] * self.speed_map[np.floor(action / 8)]
        )
        self.rect.center += mapped_action
        self.moved.append(mapped_action)

    def update(self, action=None):  # for non learning agent
        # self.rect.center += np.array([-1, 0]) # just move left
        if action:
            if isinstance(action, np.int32):
                pass
            else:
                action = action[0][0]

            mapped_action = (
                self.action_map[action % 8] * self.speed_map[np.floor(action / 8)]
            )
            self.rect.center += mapped_action
            self.moved.append(mapped_action)
        else:
            self.rect.center += np.array([-1, 0])  # just move left

    def save(self, file):
        try:
            self.model.save(file.split(".")[0])
        except:
            pass

        save_dict = {
            "direction": self.direction,
            "colour": self.colour,
            "moved": self.moved,
            "init_pos": self.init_pos,
            "objective": self.objective,
        }
        with open(file, "wb+") as filePath:
            pickle.dump(save_dict, filePath)

    def load(self, file):
        with open(file, "rb") as filePath:
            load_dict = pickle.load(filePath)
        self.direction = load_dict["direction"]
        self.colour = load_dict["colour"]
        self.moved = load_dict["moved"]
        # self.model = load_dict["model"]
        self.init_pos = load_dict["init_pos"]
        self.objective = load_dict["objective"]
        self.replay = True

    @classmethod
    def init_from_scenario(
        cls, scenario: str, window_size, *groups: AbstractGroup, **kwargs
    ):
        if scenario == ("RL_left" or "SRL_left"):
            return cls(
                window_size,
                "car",
                "left",
                *groups,
                headless=kwargs["headless"],
                colour=kwargs["colour"],
            )
        else:
            return cls(
                window_size,
                "car",
                "right",
                *groups,
                headless=kwargs["headless"],
                colour=kwargs["colour"],
            )


class RandomVehicle(Vehicle):
    def __init__(
        self,
        window_size,
        vehicleClass: str,
        direction: str,
        headless=True,
        *groups: AbstractGroup,
    ) -> None:
        if direction == "right":
            start = [0, window_size[1] / 2]
            start_offset = 100
        elif direction == "left":
            start = [window_size[0], window_size[1] / 2]
            start_offset = -100
        else:
            raise AttributeError(
                'Only "left" and "right" directions are implemented for vehicles'
            )

        super().__init__(
            start[0], start[1], 2, vehicleClass, direction, *groups, headless=headless
        )

    def update(self):
        direction = self.get_direction()

        if direction == "up":
            self.polarRandom(0, -self.speed)
        elif direction == "down":
            self.polarRandom(0, self.speed)
        elif direction == "left":
            self.polarRandom(-self.speed, 0)
        elif direction == "right":
            self.polarRandom(self.speed, 0)
        if not self.headless:
            self.animate(direction)

    def act(self, action):
        self.update()

    def polarRandom(self, dx, dy):
        epsilon = ((np.random.rand() * 2) - 1) / 200
        # epsilon = 0

        r = np.sqrt(((self.rect.x + dx) ** 2 + (self.rect.y + dy) ** 2))
        theta = np.arctan((self.rect.y + dy) / (self.rect.x + dx))

        self.rect.x = np.round(r * np.cos(theta + epsilon))
        self.rect.y = np.round(r * np.sin(theta + epsilon))


class KeyboardVehicle(Vehicle):
    def __init__(
        self,
        window_size,
        vehicleClass: str,
        direction: str,
        load_path=None,
        colour=None,
        *groups: AbstractGroup,
    ) -> None:
        self.direction = direction
        if self.direction == "right":
            start = [0, window_size[1] / 2]
            start_offset = 100
        elif self.direction == "left":
            start = [window_size[0], window_size[1] / 2]
            start_offset = -100
        else:
            raise AttributeError(
                'Only "left" and "right" directions are implemented for vehicles'
            )

        colours = [
            "blue",
            "silver",
            "white",
            "red",
            "yellow",
            "green",
            "brown",
            "orange",
        ]
        if colour is None:
            colour = np.random.choice(colours)
        else:
            pass

        super().__init__(
            start[0],
            start[1],
            2,
            vehicleClass,
            direction,
            *groups,
            headless=False,
            colour=colour,
        )
        self.rect.x += start_offset
        self.init_pos = [self.rect.x, self.rect.y]
        self.moved = [np.array([0, 0])]

        # here objective is just a place holder variable
        # to allow env to work with no RL vehicle
        self.objective = np.array(np.array([0, 0]))

        self.replay = False
        if load_path:
            self.load(load_path)

    def update(self, action=None):
        if action:
            # place holder need to implement reading in keyboard trajectories
            if isinstance(action, np.int32):
                pass
            else:
                action = action[0][0]

            mapped_action = (
                self.action_map[action % 8] * self.speed_map[np.floor(action / 8)]
            )
            self.rect.center += mapped_action
            self.moved.append(mapped_action)

        elif self.replay:
            try:
                action = self.actionHistory.pop(0)
                self.rect.center += action
            except:
                if self.direction == "down":
                    self.rect.y += self.speed
                    # self.animate("down")
                if self.direction == "up":
                    self.rect.y -= self.speed
                    # self.animate("up")
                if self.direction == "right":
                    self.rect.x += self.speed
                    # self.animate("right")
                if self.direction == "left":
                    self.rect.x -= self.speed
                    # self.animate("left")
        else:
            assert pygame.get_init(), "Keyboard agent cannot be used while headless"

            keys = pygame.key.get_pressed()

            # pad moved for no input steps
            if not any(keys):
                self.moved.append(np.array([0, 0]))

            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.rect.y += self.speed
                self.moved.append(np.array([0, 1]))
                # self.animate("down")
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                self.rect.y -= self.speed
                self.moved.append(np.array([0, -1]))
                # self.animate("up")
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.rect.x += self.speed
                self.moved.append(np.array([1, 0]))
                # self.animate("right")
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.rect.x -= self.speed
                self.moved.append(np.array([-1, 0]))
                # self.animate("left")

    def save(self, file):
        save_dict = {
            "direction": self.direction,
            "colour": self.colour,
            "moved": self.moved,
        }
        with open(file, "wb+") as filePath:
            pickle.dump(save_dict, filePath)

    def load(self, file):
        with open(file, "rb") as filePath:
            load_dict = pickle.load(filePath)
        self.direction = load_dict["direction"]
        # self.colour = load_dict["colour"]
        self.moved = load_dict["moved"]
        self.replay = True

    @classmethod
    def init_from_scenario(cls, scenario: str, window_size, **kwargs):
        if scenario == "H_left":
            return cls(window_size, "car", "left", **kwargs)
        else:
            return cls(window_size, "car", "right", **kwargs)


class RandomPedestrian(Pedestrian):
    def __init__(
        self, x, y, init_direction, *groups: AbstractGroup, headless=True
    ) -> None:
        super().__init__(x, y, init_direction, *groups, headless=headless)

    def update(self):
        direction = self.get_direction()

        if direction == "up":
            self.polarRandom(0, -self.speed)
        elif direction == "down":
            self.polarRandom(0, self.speed)
        elif direction == "left":
            self.polarRandom(-self.speed, 0)
        elif direction == "right":
            self.polarRandom(self.speed, 0)

        if not self.headless:
            self.animate(direction)

    def polarRandom(self, dx, dy):
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


class KeyboardPedestrian(Pedestrian):
    def __init__(self, x, y, init_direction, *groups: AbstractGroup) -> None:
        super().__init__(x, y, init_direction, *groups, headless=False)

        self.init_pos = [self.rect.x, self.rect.y]
        self.moved = [np.array([0, 0])]

    def update(self):
        assert pygame.get_init(), "Keyboard agent cannot be used while headless"

        keys = pygame.key.get_pressed()
        # pad moved for no input steps
        if not any(keys):
            self.moved.append(np.array([0, 0]))

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.rect.y += self.speed
            self.moved.append(np.array([0, 1]))
            self.animate("down")
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.rect.y -= self.speed
            self.moved.append(np.array([0, -1]))
            self.animate("up")
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.rect.x += self.speed
            self.moved.append(np.array([1, 0]))
            self.animate("right")
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.rect.x -= self.speed
            self.moved.append(np.array([-1, 0]))
            self.animate("left")

    def save(self, file):
        save_dict = {
            "direction": self.get_direction(),
            "moved": self.moved,
            "init_pos": self.init_pos,
        }
        with open(file, "wb+") as filePath:
            pickle.dump(save_dict, filePath)
