import random
import time
import threading
import pygame
import sys
import gym

from pygame.sprite import Sprite, Group, AbstractGroup
from typing import Union, Sequence
from spritesheet import SpriteSheet


class Pedestrian(Sprite):
    def __init__(self, x, y, direction, *groups: AbstractGroup) -> None:
        super().__init__(*groups)

        self.speed = 0.5
        self.direction = direction
        path = "images/bkspr01.png"
        self.sheet = SpriteSheet(path)
        self.image = self.sheet.image_at((0, 0, 50, 50))
        print(self.sheet, self.image)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def move(self):
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
        Sprite.__init__(*groups)

        speeds = {
            "car": 2.25,
            "bus": 1.8,
            "truck": 1.8,
            "bike": 2.5,
        }  # average speeds of vehicles

        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        # self.direction_number = direction_number
        self.direction = direction
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        if self.direction == "right":
            self.rect.x += self.speed  # move the vehicle
        elif self.direction == "down":
            self.rect.y += self.speed
        elif self.direction == "left":
            self.rect.x -= self.speed
        elif self.direction == "up":
            self.rect.y -= self.speed


class RLVehicle(Vehicle):
    def __init__(self, x, y, lane, vehicleClass, direction, *groups: AbstractGroup):
        super().__init__(x, y, lane, vehicleClass, direction, *groups)
        # self.crossed = 1

    def move(self):
        return super().move()


class TrafficVehicle(Vehicle):
    def __init__(self, x, y, lane, vehicleClass, direction, *groups: AbstractGroup):
        super().__init__(x, y, lane, vehicleClass, direction, *groups)

    # def overtake(self, group):
    # if lane < 1
    # pygame.sprite.spritecollide(self, group, False)


class Road(Sprite):
    def __init__(
        self, center_x, center_y, width, height, *groups: AbstractGroup
    ) -> None:
        super().__init__(*groups)


class TrafficSim(gym.Env):
    def __init__(
        self,
        sim_area=(1400, 800),
        headless=False,
        *vehicle_sprites: Union[Sprite, Sequence[Sprite]]
    ) -> None:
        super().__init__()
        # whether to render or not
        self.headless = headless
        self.screen_size = sim_area
        self.sim_area_x = sim_area[0]
        self.sim_area_y = sim_area[1]

        self.vehicles = Group()
        self.pedestrians = Group()
        self.roads = Group()

        for sprite in [*vehicle_sprites]:
            if isinstance(sprite, RLVehicle):
                self.vehicles.add(sprite)
            elif isinstance(sprite, KeyboardPedestrian):
                self.pedestrians.add(sprite)
            else:
                raise (
                    TypeError,
                    "Only controllable Vehicles or Pedestrians can be passed in",
                )

        print(self.pedestrians, self.vehicles)

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

        # background thread to generate traffic
        thread2 = threading.Thread(
            name="generateTraffic", target=self.generateTraffic, args=()
        )
        thread2.daemon = True
        thread2.start()

    def generateBackground(self):
        # Setting background image and scaling to window size
        bkgrd = pygame.Surface(self.screen_size)
        width_divisor, height_divisor = 20, 20
        tile_width, tile_height = (
            self.screen_size[0] / width_divisor,
            self.screen_size[1] / height_divisor,
        )
        bkgrd_img = pygame.image.load("images/grass/ground_grass_gen_05.png")
        bkgrd_tile = pygame.transform.scale(bkgrd_img, (tile_width, tile_height))
        for i in range(width_divisor):
            for j in range(height_divisor):
                bkgrd.blit(bkgrd_tile, (tile_width * i, tile_height * j))

        # bkgrd = pygame.transform.scale(
        #     pygame.image.load("images/intersection.png"), screen_size
        # )

        return bkgrd

    def generateRoads(self):
        bkgrd_img = pygame.image.load("images/grass/ground_grass_gen_05.png")

    # Generating vehicles in the simulation
    def generateTraffic(self):
        XMAX = self.sim_area_x
        YMAX = self.sim_area_y
        # Coordinates of start positions
        x = {
            "right": [0, 0, 0],
            "down": [(XMAX / 2) + 60, (XMAX / 2) + 30, (XMAX / 2)],
            "left": [XMAX, XMAX, XMAX],
            "up": [
                (XMAX / 2) - 110,
                (XMAX / 2) - 80,
                (XMAX / 2) - 50,
            ],
        }
        y = {
            "right": [
                (YMAX / 2) - 60,
                (YMAX / 2) - 30,
                (YMAX / 2),
            ],
            "down": [0, 0, 0],
            "left": [
                (YMAX / 2) + 90,
                (YMAX / 2) + 60,
                (YMAX / 2) + 30,
            ],
            "up": [YMAX, YMAX, YMAX],
        }

        while True:

            vehicle_type = random.randint(0, 3)
            lane_number = random.randint(0, 2)

            if random.uniform(0, 1) < 0.5:
                direction_number = 0
            else:
                direction_number = 2

            if random.uniform(0, 1) < 0.8:
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
        vehicle.move()

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
                old_x == 0.0
                or old_y == 0.0
                or old_x == self.sim_area_x
                or old_y == self.sim_area_y
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
        ped.move()

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
                old_x == 0.0
                or old_y == 0.0
                or old_x == self.sim_area_x
                or old_y == self.sim_area_y
            ):
                self.pedestrians.remove(ped)
            # move them back otherwise
            else:
                ped.rect.x = old_x
                ped.rect.y = old_y

    def step(self):
        for vehicle in self.vehicles:
            self.step_vehicle(vehicle)

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
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            # advance sim by one timestep
            self.step()

            # display the vehicles
            if not self.headless:
                pygame.display.update()


class Main:

    simulation = TrafficSim(sim_area=(1280, 720), headless=False)

    # add RL agent
    # simulation.add(RL_Vehicle(1, "bike", 3, "up"))

    # thread this? put methods in classes to thread this and the one above?
    simulation.run()


if __name__ == "__main__":
    Main()
