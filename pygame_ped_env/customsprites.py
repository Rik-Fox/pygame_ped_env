import random
import os
import pygame
from pygame.sprite import Sprite, AbstractGroup
import pygame_ped_env
from pygame_ped_env.spritesheet import SpriteSheet, SpriteStripAnim


class Pedestrian(Sprite):
    def __init__(self, x, y, direction, *groups: AbstractGroup) -> None:

        self.speed = 2

        self.init_anim(
            os.path.join(
                os.path.dirname(pygame_ped_env.__file__),
                "images/guy.png",
            )
        )
        self.set_direction(direction)

        self.image = self._anim[direction].next()

        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y

        self.moved = 0

        super().__init__(*groups)

    # def draw(self, screen):
    #     screen.blit(self.image, (self.x, self.y))

    def set_direction(self, direction):
        self._direction = direction

    def get_direction(self):
        return self._direction

    def init_anim(self, path):
        self._anim = {
            direction: SpriteStripAnim(path, strip_rect, 4, -1, True, 15)
            for direction, strip_rect in zip(
                [
                    "down",
                    "right",
                    "left",
                    "up",
                ],
                [(0, 0, 16, 25), (0, 25, 16, 25), (0, 48, 16, 25), (0, 72, 16, 25)],
            )
        }

    def animate(self, direction):
        # if self._direction == direction:
        self.image = self._anim[direction].next()

    def update(self):

        noise = (random.normalvariate(0, 1) * 2) - 1

        if self._direction == "right":
            self.rect.x += self.speed
            if -0.5 < noise < 0.5:
                self.rect.y += noise
        elif self._direction == "down":
            self.rect.y += self.speed
            if -0.5 < noise < 0.5:
                self.rect.x += noise
        elif self._direction == "left":
            self.rect.x -= self.speed
            if -0.5 < noise < 0.5:
                self.rect.y += noise
        elif self._direction == "up":
            self.rect.y -= self.speed
            if -0.5 < noise < 0.5:
                self.rect.x += noise

        # self.image = self._anim[self._direction].next()
        self.image = self.animate(self._direction)


class Vehicle(Sprite):
    def __init__(self, x, y, lane, vehicleClass, direction, *groups: AbstractGroup):

        speeds = {
            "car": 3.25,
            "bus": 1.8,
            "truck": 1.8,
            "bike": 2.5,
        }  # average speeds of vehicles

        self.lane_offset = (2 - lane) * 15
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        # self.direction_number = direction_number
        self.direction = direction
        path = os.path.join(
            os.path.dirname(pygame_ped_env.__file__),
            "images/" + direction + "/" + vehicleClass + ".png",
        )
        self.image = pygame.image.load(path)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        if self.direction == "right":
            self.rect.bottom = y - self.lane_offset
        elif self.direction == "down":
            self.rect.left = x + self.lane_offset
        elif self.direction == "left":
            self.rect.top = y + self.lane_offset + 5
            # self.rect.left -= 100
        elif self.direction == "up":
            self.rect.right = x - self.lane_offset

        self.moved = 0
        super().__init__(*groups)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def update(self):
        if self.direction == "right":
            self.rect.x += self.speed  # move the vehicle
        elif self.direction == "down":
            self.rect.y += self.speed
        elif self.direction == "left":
            self.rect.x -= self.speed
        elif self.direction == "up":
            self.rect.y -= self.speed


class TrafficVehicle(Vehicle):
    def __init__(self, x, y, lane, vehicleClass, direction, *groups: AbstractGroup):
        super().__init__(x, y, lane, vehicleClass, direction, *groups)

    # def overtake(self, group):
    # if lane < 1
    # pygame.sprite.spritecollide(self, group, False)


# class Road(Sprite):
#     def __init__(self, start, end, *groups: AbstractGroup) -> None:
#         XDIFF = np.abs(start[0] - end[0])
#         YDIFF = np.abs(start[1] - end[1])

#         # horizontal or vertical?
#         if XDIFF > YDIFF:
#             image_tile = pygame.transform.scale(
#                 pygame.image.load("../images/roads/roadEW.tga"), (100, 100)
#             )
#             image_rect = image_tile.get_rect()
#             surf = pygame.Surface((XDIFF, image_rect.h))
#             for t in range(np.floor(XDIFF / image_rect.h).astype(int) + 1):
#                 surf.blit(image_tile, (t * image_rect.w, 0))

#             self.image = surf
#             self.rect = self.image.get_rect()
#             self.rect.midleft = start

#         else:
#             image_tile = pygame.transform.scale(
#                 pygame.image.load("../images/roads/roadPLAZA.tga"), (100, 100)
#             )
#             image_rect = image_tile.get_rect()
#             surf = pygame.Surface((image_rect.w, YDIFF))
#             for t in range(np.floor(YDIFF / image_rect.h).astype(int) + 1):
#                 surf.blit(image_tile, (0, t * image_rect.h))

#             self.image = surf
#             self.rect = self.image.get_rect()
#             self.rect.midtop = start

#         super().__init__(*groups)
