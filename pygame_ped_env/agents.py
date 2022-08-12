import pygame
from pygame.sprite import Sprite, AbstractGroup
from pygame_ped_env.spritesheet import SpriteSheet

from pygame_ped_env.customsprites import Vehicle, Pedestrian

import numpy as np


class RLVehicle(Vehicle):
    # class for hooking in RL algorithms
    def __init__(self, x, y, lane, vehicleClass, direction, *groups: AbstractGroup):
        super().__init__(x, y, lane, vehicleClass, direction, *groups)

        # hook for RL algo
        self.model = None

    def act(self, action):
        x, y = self.rect.x, self.rect.y

        if action == "right":
            self.rect.x += self.speed  # move the vehicle
        elif action == "down":
            self.rect.y += self.speed
        elif action == "left":
            self.rect.x -= self.speed
        elif action == "up":
            self.rect.y -= self.speed

        dx = np.abs(self.rect.x - x)
        dy = np.abs(self.rect.y - y)
        self.moved = np.clip(np.sqrt(dx**2 + dy**2), 0, self.speed)

    def update(self):
        pass
        # self.act() random choice for action
        # return super().update()


class RandomVehicle(Vehicle):
    def __init__(self, x, y, init_direction, *groups: AbstractGroup) -> None:
        super().__init__(x, y, init_direction, *groups)

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


class RandomPedestrian(Pedestrian):
    def __init__(self, x, y, init_direction, *groups: AbstractGroup) -> None:
        super().__init__(x, y, init_direction, *groups)

    def update(self):

        direction = self.get_direction()

        def polarRandom(dx, dy):

            epsilon = ((np.random.rand() * 2) - 1) / 200
            # epsilon = 0

            r = np.sqrt(((self.rect.x + dx) ** 2 + (self.rect.y + dy) ** 2))
            theta = np.arctan(np.abs(self.rect.y + dy) / np.abs(self.rect.x + dx))

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
