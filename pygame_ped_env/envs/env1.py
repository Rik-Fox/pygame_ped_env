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
        simple_reward=True,
        speed_coefficient=1.0,
        position_coefficient=1.0,
        steering_coefficient=1.0,
    ) -> None:
        assert (
            2 <= len(controllable_sprites) <= 3
        ), "Must pass only 2 or 3 sprite entities to RL Crossing env"
        super().__init__()
        np.random.seed(seed)

        # whether to render or not
        self.headless = headless

        # whether to have shaped reward or not
        self.simple_reward = simple_reward

        # reward weightings, used to adjust for repsonse data
        self.speed_coeff = speed_coefficient
        self.pos_coeff = position_coefficient
        self.steer_coeff = steering_coefficient

        # this varible structure allows for sim_area =/= screen_size so events on edges
        # and spawn are not displayed as all display updates use screen_size
        self.screen_size = sim_area
        self.sim_area_x = sim_area[0]
        self.sim_area_y = sim_area[1]

        assert self.sim_area_x > self.sim_area_y, "simulation must be portrait"

        # 150 is offscreen pixel buffer, TODO: should be associated to agent sprite rect
        self.observation_space = gym.spaces.Box(
            low=-150, high=self.sim_area_x + 150, shape=(12,)
        )
        # (8 directions * 3 speeds) + 1 no_op action
        self.action_space = gym.spaces.Discrete(25)

        self.init_sprites = {}

        self.vehicle = GroupSingle()
        self.traffic = GroupSingle()
        self.pedestrian = GroupSingle()

        if len(controllable_sprites) == 2:
            sprite = Sprite()
            sprite.rect = pygame.Rect(0, 0, 0, 0)
            self.traffic.add(sprite)

        for i, sprite in enumerate(controllable_sprites):
            self.init_sprites[sprite] = sprite.rect.copy()
            if i == 0:
                self.vehicle.add(sprite)
            else:
                if isinstance(sprite, (RLVehicle, KeyboardVehicle)):
                    self.traffic.add(sprite)
                elif isinstance(sprite, (RandomPedestrian, KeyboardPedestrian)):
                    self.pedestrian.add(sprite)
                else:
                    raise TypeError(
                        "Only one or two vehicles and one pedestrian can be passed into this environment",
                    )

        self.roads = Group()
        self.generateRoads()

        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("PaAVI - Pedestrian Crossing Simulation")
            self.background = self.generateBackground()

        self.vehicleTypes = {0: "car", 1: "bus", 2: "truck", 3: "bike"}
        # discrete int action to relative action now converted in agent class
        # self.directionNumbers = {
        #     0: "down",
        #     1: "downleft",
        #     2: "left",
        #     3: "upleft",
        #     4: "up",
        #     5: "upright",
        #     6:"right",
        #     7:"downright",
        #     }

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

        if self.simple_reward:
            rwd, done = self.get_simple_reward(agent)

        else:
            # function for social rewards
            # speed_rwd, position_rwd, heading_rwd, collide_rwd, done = self.get_primitive_reward(agent)
            p_rwd, s_rwd, h_rwd, c_rwd, done = self.get_primitive_reward(agent)

            # linearly combine rewards
            # apply convex or concave adjustment, and then shift to (-1,0]
            rwd = (
                (np.power(p_rwd, self.pos_coeff) - 1)
                + (np.power(s_rwd, self.speed_coeff) - 1)
                + (np.power(h_rwd, self.steer_coeff) - 1)
                + c_rwd
            )

        return rwd, done

    def get_simple_reward(self, agent: RLVehicle):
        done = False
        collide_rwd = 0  # 0 no collision, -1 offroad, -infty hit pedestrian

        # if run over pedestrian, give reward reduction and end episode
        if pygame.sprite.spritecollide(agent.sprite, self.pedestrian, True):
            return -np.infty, True

        on_road = 0.01
        if not pygame.sprite.spritecollide(agent.sprite, self.roads, False):
            on_road = -0.01

        curr_d_obj = agent.sprite.dist_to_objective()
        # if we have reached the destination, give max rwd
        if (curr_d_obj == np.zeros(2)).all():
            # rwd for hitting goal
            rwd = 1000
            done = True
        else:
            rwd = -0.04 + on_road

        return rwd, done

    def get_primitive_reward(self, agent: RLVehicle):
        done = False
        collide_rwd = 0  # 0 no collision, -1 offroad, -infty hit pedestrian

        # if run over pedestrian, give reward reduction and end episode
        if pygame.sprite.spritecollide(agent.sprite, self.pedestrian, True):
            collide_rwd = -np.infty
            done = True

        if not pygame.sprite.spritecollide(agent.sprite, self.roads, False):
            collide_rwd = -1
            # elif not pygame.sprite.spritecollide(agent.sprite, self.roads.lane, False):
            # collide_rwd = -0.5

        # distance from end point
        curr_d_obj = agent.sprite.dist_to_objective()
        # if we have reached the destination, give max rwd
        if (curr_d_obj == np.zeros(2)).all():
            # rwd for hitting goal
            position_rwd = 1
            done = True
        else:
            # straight line vector from start to goal
            init_d_obj = agent.sprite.dist_to_objective(pos=agent.sprite.init_pos)

            # magnitudes of init and curr vectors to goal
            rho_init = np.hypot(init_d_obj[0], init_d_obj[1])
            rho_curr = np.hypot(curr_d_obj[0], curr_d_obj[1])

            # normalised difference in vector magnitudes
            rho_scaled = (rho_init - rho_curr) / rho_init

            # angle between stright line to goal and curent path to goal
            # in (0, pi]
            theta_diff = np.arccos(
                np.dot(init_d_obj, curr_d_obj) / (rho_init * rho_curr)
            )

            # recale to have theta in (0,1]
            theta_scaled = (np.pi - theta_diff) / np.pi

            # reward is linear combination of normed rho and theta
            # rho_scaled rewards for progression to goal (degined to be along x axis)
            # theta_scaled punishes for deviation from stright path (degined to be along y axis)
            position_rwd = (rho_scaled + theta_scaled) / 2

        # movement vector magnitude for this action
        delta_rho_1 = np.hypot(agent.sprite.moved[-1][0], agent.sprite.moved[-1][1])

        # rewarded for going as fast as possible, speed as fraction of top speed
        speed_rwd = delta_rho_1 / agent.sprite.speed

        # movement vector magnitude for previous action
        delta_rho_2 = np.hypot(agent.sprite.moved[-2][0], agent.sprite.moved[-2][1])

        # from angle between 2 vectors definition
        theta_s = np.arccos(
            np.clip(
                np.dot(agent.sprite.moved[-1], agent.sprite.moved[-2])
                / (
                    (delta_rho_1 * delta_rho_2) + 1e-15
                ),  # 1e-15 to avoid divide by zero
                -1.0,
                1.0,
            )  # clip to avoid round off error or +1e-15 taking arg out of arccos domain [-1,1]
        )
        # punishes proportionally for turning in this action
        heading_rwd = (np.pi - theta_s) / np.pi

        # catch any bad edge cases from calculations
        if any(np.isnan([position_rwd, speed_rwd, heading_rwd, collide_rwd])):
            print("primitive reward returned NaN")
            import pdb

            pdb.set_trace()

        return position_rwd, speed_rwd, heading_rwd, collide_rwd, done

    def step(self, action):

        self.vehicle.sprite.act(action)

        self.traffic.sprite.update(self.traffic_action)
        # print(self.traffic.sprite.rect.center)
        # print(self.traffic_action)

        try:
            self.pedestrian.sprite.update()
            ped_rect = self.pedestrian.sprite.rect
        except AttributeError:
            ped_rect = pygame.Rect(0, 0, 0, 0)
            # import pdb
            # pdb.set_trace()

        if not self.headless:
            self.render()

        obs = np.hstack(
            [
                [self.vehicle.sprite.rect],
                [self.traffic.sprite.rect],  # padding for single vehicle scenario
                [self.pedestrian.sprite.rect],
            ],
        )
        try:
            self.traffic_action = self.traffic.sprite.model.predict(obs)
        except:
            self.traffic_action = None

        rwd, self.done = self.get_reward(self.vehicle)

        # remove vehicles once they drive offscreen and finish episode
        if not self.done:
            if not (0 <= self.vehicle.sprite.rect.centery <= self.sim_area_y):
                # if not roughly at destination then punish
                if not (
                    self.vehicle.sprite.dist_to_objective() <= np.ones(2) * 3
                ).all():
                    rwd -= 1000
                self.vehicle.remove(self.vehicle.sprite)
                self.done = True
            elif not (0 <= self.vehicle.sprite.rect.centerx <= self.sim_area_x):
                # if not roughly at destination then punish
                if not (
                    self.vehicle.sprite.dist_to_objective() <= np.ones(2) * 3
                ).all():
                    rwd -= 1000
                self.vehicle.remove(self.vehicle.sprite)
                self.done = True

        return obs, rwd, self.done, {}

    def render(self):
        self.screen.blit(self.background, (0, 0))  # display background in simulation
        self.roads.draw(self.screen)
        self.vehicle.draw(self.screen)
        self.pedestrian.draw(self.screen)

        try:
            self.traffic.draw(self.screen)
        except:
            pass

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        pygame.display.update()

        # hacky way to have ~60 FPS TODO: implentment properly
        time.sleep(0.016)

    def reset(self):

        self.vehicle.empty()
        self.traffic.empty()
        self.pedestrian.empty()

        if len([*self.init_sprites.keys()]) == 2:
            sprite = Sprite()
            sprite.rect = pygame.Rect(0, 0, 0, 0)
            self.traffic.add(sprite)

        for i, sprite in enumerate([*self.init_sprites.keys()]):
            if isinstance(sprite, (RLVehicle, KeyboardVehicle)):
                sprite.rect = self.init_sprites[sprite].copy()
                if i == 0:
                    self.vehicle.add(sprite)
                else:
                    self.traffic.add(sprite)
            elif isinstance(sprite, (RandomPedestrian, KeyboardPedestrian)):
                sprite.rect = self.init_sprites[sprite].copy()
                # slight noise in pedestrian start position
                sprite.rect.x, sprite.rect.y = (
                    sprite.rect.x + (np.random.rand() * 4) - 2,
                    sprite.rect.y + (np.random.rand() * 4) - 2,
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

        try:
            self.traffic_action = self.traffic.sprite.model.predict(obs)
        except:
            self.traffic_action = None

        self.done = False
        obs = np.hstack(
            [
                [self.vehicle.sprite.rect],
                [self.traffic.sprite.rect],  # padding for single vehicle scenario
                [self.pedestrian.sprite.rect],
            ],
        )

        return obs

    def run(self):
        # count = 0
        obs = self.reset()
        done = self.done
        while not done:
            # advance sim by one timestep
            obs, _, done, _ = self.step(self.vehicle.sprite.model.predict(obs)[0])

            # display the simulation
            if not self.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()

                self.render()
        pygame.quit()
