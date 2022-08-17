import time
import threading
import pygame
import sys
import os
import gym
import numpy as np

from pygame.sprite import Sprite, Group, GroupSingle
from typing import Union, Sequence
import pygame_ped_env

from pygame_ped_env.agents import (
    RLVehicle,
    KeyboardVehicle,
    KeyboardPedestrian,
    RandomPedestrian,
)


class RLCrossingSim(gym.Env):
    def __init__(
        self,
        sim_area,
        controllable_sprites: Union[None, Sprite, Sequence[Sprite]] = None,
        headless=False,
        seed=1234,
    ) -> None:
        assert (
            len(controllable_sprites) == 2
        ), "Must pass only 2 sprite entities to RL Crossing env"
        super().__init__()
        np.random.seed(seed)

        # whether to render or not
        self.headless = headless

        # this varible structure allows for sim_area =/= screen_size so events on edges
        # and spawn are not displayed as all display updates use screen_size
        self.screen_size = sim_area
        self.sim_area_x = sim_area[0]
        self.sim_area_y = sim_area[1]

        assert self.sim_area_x > self.sim_area_y, "simulation must be portrait"

        # 150 is offscreen pixel buffer, TODO: should be associated to agent sprite rect
        self.observation_space = gym.spaces.Box(
            low=-150, high=self.sim_area_x + 150, shape=(8,)
        )
        self.action_space = gym.spaces.Discrete(4)

        self.init_sprites = {}

        for sprite in controllable_sprites:
            if isinstance(sprite, (RLVehicle, KeyboardVehicle)):
                self.vehicle = GroupSingle(sprite)
                self.init_sprites[sprite] = sprite.rect.copy()
            elif isinstance(sprite, (RandomPedestrian, KeyboardPedestrian)):
                self.pedestrian = GroupSingle(sprite)
                self.init_sprites[sprite] = sprite.rect.copy()
            else:
                raise TypeError(
                    "Only one vehicle and one pedestrian can be passed into environment",
                )

        self.roads = Group()
        self.generateRoads()

        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("PaAVI - Pedestrian Crossing Simulation")
            self.background = self.generateBackground()

        self.vehicleTypes = {0: "car", 1: "bus", 2: "truck", 3: "bike"}
        self.directionNumbers = {0: "down", 1: "left", 2: "up", 3: "right"}

        # Gap between vehicles
        # movingGap = 15  # moving gap
        # self.spacings = {0: -movingGap, 1: -movingGap, 2: movingGap, 3: movingGap}

        self.done = False

    def generateBackground(self):
        # Setting background image and scaling to window size
        bkgrd = pygame.Surface(self.screen_size)
        width_divisor, height_divisor = 2, 2
        tile_width, tile_height = (
            self.screen_size[0] / width_divisor,
            self.screen_size[1] / height_divisor,
        )
        bkgrd_img = pygame.image.load(
            os.path.join(
                os.path.dirname(pygame_ped_env.__file__),
                "images/grass/ground_grass_gen_02.png",
            )
        ).convert()
        # bkgrd = pygame.transform.scale(bkgrd_img, (self.screen_size))
        bkgrd_tile = pygame.transform.scale(bkgrd_img, (tile_width, tile_height))
        for i in range(width_divisor):
            for j in range(height_divisor):
                bkgrd.blit(bkgrd_tile, (tile_width * i, tile_height * j))

        # bkgrd = pygame.transform.scale(
        #     pygame.image.load("../images/intersection.png"), screen_size
        # )

        return bkgrd

    def generateRoads(self):
        scale = (self.sim_area_x / 5, self.sim_area_y / 5)
        # fairly ineffecient TODO: refactor to road class for generality

        vertical_path = Sprite(self.roads)
        image = pygame.transform.scale(
            pygame.image.load(
                os.path.join(
                    os.path.dirname(pygame_ped_env.__file__),
                    "images/roads/roadPLAZA.tga",
                )
            ),
            scale,
        )
        image_tile = pygame.Surface((scale[0] / 2, scale[1] / 2))
        image_tile.blit(image, (0, 0), image_tile.get_rect())
        image_rect = image_tile.get_rect()
        surf = pygame.Surface((image_rect.w, self.sim_area_y))
        for t in range(np.floor(self.sim_area_y / image_rect.h).astype(int) + 1):
            surf.blit(image_tile, (0, t * image_rect.h))

        vertical_path.image = surf
        vertical_path.rect = vertical_path.image.get_rect()
        vertical_path.rect.midtop = (self.sim_area_x / 2, 0)

        horizontal_road = Sprite(self.roads)
        image_tile = pygame.transform.scale(
            pygame.image.load(
                os.path.join(
                    os.path.dirname(pygame_ped_env.__file__),
                    "images/roads/roadEW.tga",
                )
            ),
            scale,
        )
        image_rect = image_tile.get_rect()
        surf = pygame.Surface((self.sim_area_x, image_rect.h))
        for t in range(np.floor(self.sim_area_x / image_rect.h).astype(int) + 1):
            surf.blit(image_tile, (t * image_rect.w, 0))

        horizontal_road.image = surf
        horizontal_road.rect = horizontal_road.image.get_rect()
        horizontal_road.rect.midleft = (0, self.sim_area_y / 2)

    def get_reward(self, agent: RLVehicle):
        # place holder function for social rewards
        return self.get_primitive_reward(agent)

    def get_primitive_reward(self, agent: RLVehicle):
        # agent = self.vehicle.sprite
        # if run over pedestrian, give massive reward reduction and end episode
        done = False
        c = 1
        if pygame.sprite.spritecollide(agent, self.pedestrian, True):
            c = 0
            done = True

        # distance from end point
        d_obj = np.abs(agent.rect.center - agent.objective)
        d_theta = np.arctan(d_obj[1] / d_obj[2])
        # movement vector
        delta_rho = np.sqrt(agent.sprite.moved[0] ** 2 + agent.sprite.moved[1] ** 2)
        delta_theta = np.arctan(agent.sprite.moved[1] / agent.sprite.moved[0])

        # primitive reward function
        rwd = (delta_theta / d_theta) * (delta_rho / agent.sprite.speed) * ((2 * c) + 1)

        return rwd, done

    def step(self, action):
        # print(action)
        self.vehicle.sprite.act(self.directionNumbers[action])

        self.pedestrian.sprite.update()

        if not self.headless:
            self.render()

        obs = np.hstack(
            [
                [self.vehicle.sprite.rect],
                [self.pedestrian.sprite.rect],
            ],
        )
        rwd, self.done = self.get_reward(self.vehicle)

        # remove vehicles once they drive offscreen and finish episode
        if not self.done:
            if not (
                -self.vehicle.sprite.rect.width
                <= self.vehicle.sprite.rect.x
                <= self.sim_area_x + self.vehicle.sprite.rect.width
            ):
                self.vehicle.remove(self.vehicle.sprite)
                self.done = True
            elif not (
                -self.vehicle.sprite.rect.height
                <= self.vehicle.sprite.rect.y
                <= self.sim_area_y + self.vehicle.sprite.rect.height
            ):
                self.vehicle.remove(self.vehicle.sprite)
                self.done = True

        return obs, rwd, self.done, {}

    def render(self):
        self.screen.blit(self.background, (0, 0))  # display background in simulation
        self.roads.draw(self.screen)
        self.vehicle.draw(self.screen)
        self.pedestrian.draw(self.screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        pygame.display.update()

        # hacky way to have ~60 FPS TODO: implentment properly
        time.sleep(0.016)

    def reset(self):

        self.vehicle.empty()
        self.pedestrian.empty()

        for sprite in [*self.init_sprites.keys()]:
            if isinstance(sprite, (RLVehicle, KeyboardVehicle)):
                sprite.rect = self.init_sprites[sprite].copy()
                self.vehicle.add(sprite)
            elif isinstance(sprite, (RandomPedestrian, KeyboardPedestrian)):
                sprite.rect = self.init_sprites[sprite].copy()
                # slight noise in pedestrian start position
                sprite.rect.x, sprite.rect.y = (
                    np.random.rand() * 5,
                    np.random.rand() * 5,
                )
                self.pedestrian.add(sprite)
            else:
                raise TypeError(
                    "Only controllable Vehicles or Pedestrians can be passed into environment",
                )

        if not self.headless:
            # pygame.quit()
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("PaAVI - Pedestrian Crossing Simulation")
            self.background = self.generateBackground()

        self.done = False
        obs = np.hstack(
            [
                [self.vehicle.sprite.rect],
                [self.pedestrian.sprite.rect],
            ],
        )

        return obs

    def run(self):
        # count = 0
        while True:
            # advance sim by one timestep
            self.step()
            # count += 1
            # if count % 100000 == 0:
            #     print(count)
            # randPed = [
            #     ped for ped in self.pedestrians if isinstance(ped, RandomPedestrian)
            # ]
            # print(randPed)
            # if len(randPed) < 1:
            #     RandomPedestrian(
            #         self.sim_area_x / 2,
            #         self.sim_area_y * (3 / 4),
            #         "up",
            #         self.pedestrians,
            #     ),

            # display the simulation
            if not self.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()

                pygame.display.update()

                # hacky way to have ~60 FPS TODO: implentment properly
                time.sleep(0.016)