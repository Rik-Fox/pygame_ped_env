from lib2to3.pgen2 import driver
import random
import time
import threading
import pygame
import sys

from pygame.sprite import Sprite, Group
from typing import Union, Sequence


class Vehicle(Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction):
        Sprite.__init__(self)

        speeds = {
            "car": 2.25,
            "bus": 1.8,
            "truck": 1.8,
            "bike": 2.5,
        }  # average speeds of vehicles

        # Coordinates of vehicles' start
        x = {
            "right": [0, 0, 0],
            "down": [755, 727, 697],
            "left": [1400, 1400, 1400],
            "up": [602, 627, 657],
        }
        y = {
            "right": [348, 370, 398],
            "down": [0, 0, 0],
            "left": [498, 466, 436],
            "up": [800, 800, 800],
        }

        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.crossed = 0
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)
        self.rect = self.image.get_rect()
        self.rect.x = x[direction][lane]
        self.rect.y = y[direction][lane]

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


class RL_Vehicle(Vehicle):
    def __init__(self, lane, vehicleClass, direction_number, direction):
        super().__init__(lane, vehicleClass, direction_number, direction)
        # self.crossed = 1


class Traffic_Vehicle(Vehicle):
    def __init__(self, lane, vehicleClass, direction_number, direction):
        super().__init__(lane, vehicleClass, direction_number, direction)

    # def overtake(self, group):
    # if lane < 1
    # pygame.sprite.spritecollide(self, group, False)


class TrafficSim(Group):
    def __init__(
        self, screen_size, headless=False, *sprites: Union[Sprite, Sequence[Sprite]]
    ) -> None:
        super().__init__(*sprites)
        # whether to render or not
        self.headless = headless
        self.screen_width = screen_size[0]
        self.screen_height = screen_size[1]

        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode(screen_size)
            # self.screen_width = self.screen.get_width()
            # self.screen_height = self.screen.get_height()
            pygame.display.set_caption("PaAVI - Pedestrian Crossing Simulation")

            # Setting background image i.e. image of intersection
            self.background = pygame.image.load("images/intersection.png")

        self.vehicleTypes = {0: "car", 1: "bus", 2: "truck", 3: "bike"}
        self.directionNumbers = {0: "right", 1: "down", 2: "left", 3: "up"}

        # Gap between vehicles
        # movingGap = 15  # moving gap
        # self.spacings = {0: -movingGap, 1: -movingGap, 2: movingGap, 3: movingGap}

    # Generating vehicles in the simulation
    def generateVehicles(self):
        while True:
            vehicle_type = random.randint(0, 3)
            lane_number = random.randint(1, 2)

            # lane_number = 1
            # dist = [25, 50, 75, 100]
            # if temp < dist[0]:
            #     direction_number = 0
            # elif temp < dist[1]:
            #     direction_number = 1
            # elif temp < dist[2]:
            #     direction_number = 2
            # elif temp < dist[3]:
            #     direction_number = 3
            # dist = 50

            if random.uniform(0, 1) < 0.5:
                direction_number = 0
            else:
                direction_number = 2

            self.add(
                Vehicle(
                    lane_number,
                    self.vehicleTypes[vehicle_type],
                    direction_number,
                    self.directionNumbers[direction_number],
                )
            )
            # self.step_vehicle([*self.spritedict.keys()][-1])
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
            <= self.screen_width + vehicle.rect.width
        ):
            self.remove(vehicle)
        if not (
            -vehicle.rect.height
            <= vehicle.rect.y
            <= self.screen_height + vehicle.rect.height
        ):
            self.remove(vehicle)

        # check if vehicle has collided with another vehicle
        if len(pygame.sprite.spritecollide(vehicle, self, False)) > 1:

            # destroy them if collision at spawn, as this logic only works
            # if the collision is strictly caused by this vehicles updated position
            if (
                old_x == 0.0
                or old_y == 0.0
                or old_x == self.screen_width
                or old_y == self.screen_width
            ):
                # pygame.sprite.spritecollide(vehicle, simulation, True)
                self.remove(vehicle)
            # move them back otherwise
            else:
                vehicle.rect.x = old_x
                vehicle.rect.y = old_y

    def step(self):
        for vehicle in self:
            self.step_vehicle(vehicle)

        if not self.headless:
            self.screen.blit(
                self.background, (0, 0)
            )  # display background in simulation
            self.render()

    def render(self):
        self.draw(self.screen)

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

    simulation = TrafficSim(screen_size=(1400, 800), headless=False)

    # add RL agent
    # simulation.add(RL_Vehicle(1, "bike", 3, "up"))

    # background thread to generate traffic
    thread2 = threading.Thread(
        name="generateVehicles", target=simulation.generateVehicles, args=()
    )
    thread2.daemon = True
    thread2.start()

    # thread this? put methods in classes to thread this and the one above?
    simulation.run()


if __name__ == "__main__":
    Main()
