import pygame
from pygame.sprite import Sprite, AbstractGroup
from spritesheet import SpriteSheet


class Pedestrian(Sprite):
    def __init__(self, x, y, direction, *groups: AbstractGroup) -> None:

        self.speed = 1
        self.direction = direction
        path = "images/16x24rpgdb.gif"
        self.sheet = SpriteSheet(path)
        self.image = self.sheet.image_at((30, 50, 10, 10))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        super().__init__(*groups)

    def update(self):
        if self.direction == "right":
            self.rect.x += self.speed
        elif self.direction == "down":
            self.rect.y += self.speed
        elif self.direction == "left":
            self.rect.x -= self.speed
        elif self.direction == "up":
            self.rect.y -= self.speed


class KeyboardPedestrian(Pedestrian):
    def __init__(self, x, y, lane, direction, *groups: AbstractGroup) -> None:
        super().__init__(x, y, lane, direction, *groups)


class Vehicle(Sprite):
    def __init__(self, x, y, lane, vehicleClass, direction, *groups: AbstractGroup):

        speeds = {
            "car": 2.25,
            "bus": 1.8,
            "truck": 1.8,
            "bike": 2.5,
        }  # average speeds of vehicles

        self.lane = lane
        LANEWIDTH = 15
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        # self.direction_number = direction_number
        self.direction = direction
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        if self.direction == "right":
            self.rect.bottom = y - ((2 - self.lane) * LANEWIDTH)
        elif self.direction == "down":
            self.rect.left = x + ((2 - self.lane) * LANEWIDTH)
        elif self.direction == "left":
            self.rect.top = y + ((2 - self.lane) * LANEWIDTH)
        elif self.direction == "up":
            self.rect.right = x - ((2 - self.lane) * LANEWIDTH)

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


class RLVehicle(Vehicle):
    # class for hooking in RL algorithms
    def __init__(self, x, y, lane, vehicleClass, direction, *groups: AbstractGroup):
        super().__init__(x, y, lane, vehicleClass, direction, *groups)
        # self.crossed = 1

    def act(self):
        return super().update()


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
#                 pygame.image.load("images/roads/roadEW.tga"), (100, 100)
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
#                 pygame.image.load("images/roads/roadPLAZA.tga"), (100, 100)
#             )
#             image_rect = image_tile.get_rect()
#             surf = pygame.Surface((image_rect.w, YDIFF))
#             for t in range(np.floor(YDIFF / image_rect.h).astype(int) + 1):
#                 surf.blit(image_tile, (0, t * image_rect.h))

#             self.image = surf
#             self.rect = self.image.get_rect()
#             self.rect.midtop = start

#         super().__init__(*groups)
