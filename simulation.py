import random
import time
import threading
import pygame
import sys
import gym
import numpy as np

from pygame.sprite import Sprite, Group
from typing import Union, Sequence

from customsprites import Vehicle, RLVehicle, Pedestrian, KeyboardPedestrian  # , Road


class TrafficSim(gym.Env):
    def __init__(
        self,
        sim_area,
        *controllable_sprites: Union[Sprite, Sequence[Sprite]],
        headless=False,
        traffic=False,
    ) -> None:
        super().__init__()
        # whether to render or not
        self.headless = headless

        # this varible structure allows for sim_area =/= screen_size so events on edges
        # and spawn are not displayed as all display updates use screen_size
        self.screen_size = sim_area
        self.sim_area_x = sim_area[0]
        self.sim_area_y = sim_area[1]

        self.vehicles = Group()
        self.pedestrians = Group()
        self.roads = Group()

        for sprite in [*controllable_sprites]:
            if isinstance(sprite, RLVehicle):
                self.vehicles.add(sprite)
            elif isinstance(sprite, KeyboardPedestrian):
                self.pedestrians.add(sprite)
            else:
                raise (
                    TypeError,
                    "Only controllable Vehicles or Pedestrians can be passed into environment",
                )

        self.generateRoads()

        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("PaAVI - Pedestrian Crossing Simulation")
            self.background = self.generateBackground()

        self.vehicleTypes = {0: "car", 1: "bus", 2: "truck", 3: "bike"}
        self.directionNumbers = {0: "right", 1: "down", 2: "left", 3: "up"}

        # Gap between vehicles
        # movingGap = 15  # moving gap
        # self.spacings = {0: -movingGap, 1: -movingGap, 2: movingGap, 3: movingGap}
        if traffic:
            # background thread to generate traffic
            thread2 = threading.Thread(
                name="generateTraffic", target=self.generateTraffic, args=()
            )
            thread2.daemon = True
            thread2.start()

    def generateBackground(self):
        # Setting background image and scaling to window size
        bkgrd = pygame.Surface(self.screen_size)
        width_divisor, height_divisor = 2, 2
        tile_width, tile_height = (
            self.screen_size[0] / width_divisor,
            self.screen_size[1] / height_divisor,
        )
        bkgrd_img = pygame.image.load("images/grass/ground_grass_gen_02.png").convert()
        # bkgrd = pygame.transform.scale(bkgrd_img, (self.screen_size))
        bkgrd_tile = pygame.transform.scale(bkgrd_img, (tile_width, tile_height))
        for i in range(width_divisor):
            for j in range(height_divisor):
                bkgrd.blit(bkgrd_tile, (tile_width * i, tile_height * j))

        # bkgrd = pygame.transform.scale(
        #     pygame.image.load("images/intersection.png"), screen_size
        # )

        return bkgrd

    def generateRoads(self):
        # fairly ineffecient TODO: refactor to road class for generality

        horizontal_road = Sprite(self.roads)
        image_tile = pygame.transform.scale(
            pygame.image.load("images/roads/roadEW.tga"), (100, 100)
        )
        image_rect = image_tile.get_rect()
        surf = pygame.Surface((self.sim_area_x, image_rect.h))
        for t in range(np.floor(self.sim_area_x / image_rect.h).astype(int) + 1):
            surf.blit(image_tile, (t * image_rect.w, 0))

        horizontal_road.image = surf
        horizontal_road.rect = horizontal_road.image.get_rect()
        horizontal_road.rect.midleft = (0, self.sim_area_y / 2)

        vertical_path = Sprite(self.roads)
        image = pygame.transform.scale(
            pygame.image.load("images/roads/roadPLAZA.tga"), (100, 100)
        )
        image_tile = pygame.Surface((50, 50))
        image_tile.blit(image, (0, 0), image_tile.get_rect())
        image_rect = image_tile.get_rect()
        surf = pygame.Surface((image_rect.w, self.sim_area_y))
        for t in range(np.floor(self.sim_area_y / image_rect.h).astype(int) + 1):
            surf.blit(image_tile, (0, t * image_rect.h))

        vertical_path.image = surf
        vertical_path.rect = vertical_path.image.get_rect()
        vertical_path.rect.midtop = (self.sim_area_x / 2, 0)

        self.crossing = Sprite(self.roads)
        self.crossing.image = pygame.transform.scale(
            pygame.image.load("images/roads/roadNEWS.tga"), (100, 100)
        )
        self.crossing.rect = self.crossing.image.get_rect()
        self.crossing.rect.center = ((self.sim_area_x / 2), (self.sim_area_y / 2))

    # Generating vehicles in the simulation
    def generateTraffic(self):
        # Coordinates of start positions
        x = {
            "right": [0, 0, 0],
            "down": [
                (self.sim_area_x / 2),
                (self.sim_area_x / 2),
                (self.sim_area_x / 2),
            ],
            "left": [self.sim_area_x, self.sim_area_x, self.sim_area_x],
            "up": [
                (self.sim_area_x / 2),
                (self.sim_area_x / 2),
                (self.sim_area_x / 2),
            ],
        }
        y = {
            "right": [
                (self.sim_area_y / 2),
                (self.sim_area_y / 2),
                (self.sim_area_y / 2),
            ],
            "down": [0, 0, 0],
            "left": [
                (self.sim_area_y / 2),
                (self.sim_area_y / 2),
                (self.sim_area_y / 2),
            ],
            "up": [self.sim_area_y, self.sim_area_y, self.sim_area_y],
        }

        while True:

            vehicle_type = random.randint(0, 3)
            # lane_number = random.randint(0, 2)
            lane_number = 2

            if random.uniform(0, 1) < 0.5:
                direction_number = 0
            else:
                direction_number = 2

            if random.uniform(0, 1) < 0.7:
                direction = self.directionNumbers[direction_number]
                Vehicle(
                    x[direction][lane_number],
                    y[direction][lane_number],
                    lane_number,
                    self.vehicleTypes[vehicle_type],
                    direction,
                    self.vehicles,
                )
            else:
                direction_number += 1
                direction = self.directionNumbers[direction_number]
                Pedestrian(
                    x[direction][lane_number],
                    y[direction][lane_number],
                    direction,
                    self.pedestrians,
                )

            time.sleep(0.2)

    def step_vehicle(self, vehicle):
        # save original vehicle position
        old_x = vehicle.rect.x
        old_y = vehicle.rect.y

        # update vehicle position
        vehicle.update()

        # remove vehicles once they drive offscreen
        if not (
            -vehicle.rect.width
            <= vehicle.rect.x
            <= self.sim_area_x + vehicle.rect.width
        ):
            self.vehicles.remove(vehicle)
        if not (
            -vehicle.rect.height
            <= vehicle.rect.y
            <= self.sim_area_y + vehicle.rect.height
        ):
            self.vehicles.remove(vehicle)

        # check if vehicle has collided with another vehicle
        if len(pygame.sprite.spritecollide(vehicle, self.vehicles, False)) > 1:

            # destroy them if collision at spawn, as this logic only works
            # if the collision is strictly caused by this vehicles updated position
            if (
                old_x <= 0.0
                or old_y <= 0.0
                or old_x >= self.sim_area_x
                or old_y >= self.sim_area_y
            ):
                self.vehicles.remove(vehicle)
            # move them back otherwise
            else:
                vehicle.rect.x = old_x
                vehicle.rect.y = old_y

    def step_pedestrian(self, ped):
        # save original vehicle position
        old_x = ped.rect.x
        old_y = ped.rect.y

        # update vehicle position
        ped.update()

        # remove vehicles once they drive offscreen
        if not (-ped.rect.width <= ped.rect.x <= self.sim_area_x + ped.rect.width):
            self.pedestrians.remove(ped)
        if not (-ped.rect.height <= ped.rect.y <= self.sim_area_y + ped.rect.height):
            self.pedestrians.remove(ped)

        # check if vehicle has collided with another vehicle
        if len(pygame.sprite.spritecollide(ped, self.pedestrians, False)) > 1:

            # destroy them if collision at spawn, as this logic only works
            # if the collision is strictly caused by this vehicles updated position
            if (
                old_x <= 0.0
                or old_y <= 0.0
                or old_x >= self.sim_area_x
                or old_y >= self.sim_area_y
            ):
                self.pedestrians.remove(ped)
            # move them back otherwise
            else:
                ped.rect.x = old_x
                ped.rect.y = old_y

    def step(self):
        for vehicle in self.vehicles:
            self.step_vehicle(vehicle)
        for ped in self.pedestrians:
            self.step_pedestrian(ped)

        if not self.headless:
            self.screen.blit(
                self.background, (0, 0)
            )  # display background in simulation
            self.render()

    def render(self):
        self.roads.draw(self.screen)
        self.vehicles.draw(self.screen)
        self.pedestrians.draw(self.screen)

    def run(self):
        # count = 0
        while True:
            # advance sim by one timestep
            self.step()
            # count += 1
            # if count % 100000 == 0:
            #     print(count)

            # display the simulation
            if not self.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()

                pygame.display.update()
                # hacky way to have ~60 FPS TODO: implentment properly
                time.sleep(0.016)


class Main:
    window = (720, 576)
    simulation = TrafficSim(
        window,
        RLVehicle(0, window[1] / 2, 2, "car", "right"),
        headless=False,
        traffic=True,
    )

    # RL agent
    # RL_Vehicle(window[0]/2, window[1], 2, "car", "up")

    simulation.run()


if __name__ == "__main__":
    Main()
