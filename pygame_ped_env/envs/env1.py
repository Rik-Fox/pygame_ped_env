import time
import pygame
import sys
import os
import gym
import numpy as np

from stable_baselines3 import DQN
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
        scenarioList: Union[None, Sequence[int]] = None,
        human_controlled_ped=False,
        human_controlled_car=False,
        headless=False,
        seed=1234,
        basic_model: any = None,
        attr_model: any = None,
        log_path=None,
        speed_coefficient=1.0,
        position_coefficient=1.0,
        steering_coefficient=1.0,
    ) -> None:

        super().__init__()
        np.random.seed(seed)

        self.log_path = log_path

        # whether to render or not
        self.headless = headless

        # whether to have shaped reward or not
        # true by default, scenario will change it in reset()
        self.simple_reward = True

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
        self.action_space.seed(seed)

        init_sprite = Sprite()
        init_sprite.model = None
        init_sprite.rect = pygame.Rect(0, 0, 0, 0)

        self.vehicle = GroupSingle(init_sprite)
        self.traffic = GroupSingle(init_sprite)
        self.pedestrian = GroupSingle(init_sprite)

        if scenarioList is not None:
            self.scenarioList = scenarioList
        else:
            self.scenarioList = [0]

        self.scenario = lambda: np.random.choice(self.scenarioList)

        self.roads = Group()
        self.generateRoads()

        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("PaAVI - Pedestrian Crossing Simulation")
            self.background = self.generateBackground()

        if not basic_model:
            modelB_path = os.path.join(
                os.path.dirname(self.log_path), "simple_reward_agent", "init_model"
            )
            try:
                self.modelB = DQN.load(modelB_path, env=self)
            except:
                basic_model = DQN(
                    "MlpPolicy",
                    self,
                    verbose=0,
                    tensorboard_log=os.path.join(
                        os.path.dirname(self.log_path), "simple_reward_agent"
                    ),
                )
                basic_model._setup_model()
                basic_model.save(modelB_path, include="env")

                self.modelB = DQN.load(modelB_path, env=self)
        else:
            self.modelB = DQN.load(basic_model, env=self)

        if not attr_model:
            modelA_path = os.path.join(
                os.path.dirname(self.log_path), "shaped_reward_agent", "init_model"
            )
            try:
                self.modelA = DQN.load(modelA_path, env=self)
            except:
                attr_model = DQN(
                    "MlpPolicy",
                    self,
                    verbose=0,
                    tensorboard_log=os.path.join(
                        os.path.dirname(self.log_path), "shaped_reward_agent"
                    ),
                )
                attr_model.save(modelA_path, include="env")
                self.modelA = DQN.load(modelA_path, env=self)
        else:
            self.modelA = DQN.load(attr_model, env=self)

        self.training = not human_controlled_ped

        self.H_collect = human_controlled_car

    def get_max_reward(self, simple_reward=False):
        # highest possible reward for both cases
        if simple_reward:
            return (0.03 * (self.sim_area_x / 3)) * 2
        else:
            return np.sum(np.array(range(0, self.sim_area_x, 3)) / self.sim_area_x) * 2

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
            rwd = self.get_max_reward(self.simple_reward)
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
            position_rwd = self.get_max_reward(self.simple_reward)
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
        # self.vehicle.sprite.act([[6], None]) # just move right for testing without trained model

        self.traffic.sprite.update(self.traffic_action)

        # add placeholder ped if its gone offscreen and been removed
        try:
            self.pedestrian.sprite.update()
        except:
            self.pedestrian.add(Sprite())
            self.pedestrian.sprite.rect = pygame.Rect(0, 0, 0, 0)

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
            veh_rect = self.vehicle.sprite.rect
            if not (0 - veh_rect.h <= veh_rect.centery <= self.sim_area_y + veh_rect.h):
                # if not roughly at destination then punish
                if not (
                    self.vehicle.sprite.dist_to_objective() <= np.ones(2) * 3
                ).all():
                    rwd -= 3000
                self.done = True
            elif not (
                0 - veh_rect.w <= veh_rect.centerx <= self.sim_area_x + veh_rect.w
            ):
                # if not roughly at destination then punish
                if not (
                    self.vehicle.sprite.dist_to_objective() <= np.ones(2) * 3
                ).all():
                    rwd -= 3000
                self.done = True

            # if vehicle not done, check for ped offscreen
            if not self.done:
                ped_rect = self.pedestrian.sprite.rect
                if not (
                    0 - ped_rect.h <= ped_rect.centery <= self.sim_area_y + ped_rect.h
                ):
                    self.pedestrian.remove(self.pedestrian.sprite)
                elif not (
                    0 - ped_rect.w <= ped_rect.centerx <= self.sim_area_y + ped_rect.w
                ):
                    self.pedestrian.remove(self.pedestrian.sprite)
        else:
            try:
                if self.traffic.sprite.replay:
                    self.traffic.sprite.save(
                        os.path.join(
                            self.log_path,
                            f"{self.scenarioName}_human_trajectory_{np.random.randint(5)}",
                        )
                    )
            except:
                pass
        return obs, rwd, self.done, {}

    def render(self):
        self.screen.blit(self.background, (0, 0))  # display background in simulation
        self.roads.draw(self.screen)
        self.vehicle.draw(self.screen)

        try:
            self.pedestrian.draw(self.screen)
        except:
            pass

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
        scenario = self.scenario_map()[self.scenario()]
        self.scenarioName = scenario["name"]

        self.vehicle.empty()
        agent = scenario["agentSprites"][0]
        if agent[0] == "R":
            agentL = RLVehicle.init_from_scenario(agent, self.screen_size)
            agentL.model = self.modelA
            self.simple_reward = False
        elif agent[0] == "S":
            agentL = RLVehicle.init_from_scenario(agent[1:], self.screen_size)
            agentL.model = self.modelB
            self.simple_reward = True
        else:
            if self.training:
                load_path = os.path.join(
                    self.log_path,
                    f"{self.scenarioName}_human_trajectory_{np.random.randint(5)}",
                )
            agentL = KeyboardVehicle.init_from_scenario(
                agent, self.screen_size, load_path=None
            )
            agentL.model = None

        self.modelL = agentL.model
        self.vehicle.add(agentL)

        self.traffic.empty()
        try:
            traffic_agent = scenario["agentSprites"][1]

            if traffic_agent[0] == "R":
                agentT = RLVehicle.init_from_scenario(traffic_agent, self.screen_size)
                agentT.model = self.modelA
            elif traffic_agent[0] == "S":
                agentT = RLVehicle.init_from_scenario(
                    traffic_agent[1:], self.screen_size
                )
                agentT.model = self.modelB
            else:
                agentT = KeyboardVehicle.init_from_scenario(
                    traffic_agent, self.screen_size
                )
                agentT.model = None
        except:
            agentT = Sprite()
            agentT.rect = pygame.Rect(0, 0, 0, 0)
            agentT.model = None

        self.modelT = agentT.model
        self.traffic.add(agentT)

        # add pedestrian, keyboard for experiment or random if just training or code evalling
        self.pedestrian.empty()
        if self.training or self.H_collect:
            self.pedestrian.add(
                RandomPedestrian(
                    # add slight noise in pedestrian start position
                    (self.screen_size[0] / 2) + (np.random.rand() * 4) - 2,
                    (self.screen_size[1] * (7 / 8)) + (np.random.rand() * 4) - 2,
                    "up",
                )
            )
        else:
            self.pedestrian.add(
                KeyboardPedestrian(
                    self.screen_size[0] / 2, self.screen_size[1] * (7 / 8), "up"
                )
            )

        if not self.headless:
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

    @classmethod
    def init_scenario(cls, scenario, sim_area, training=False, **kwargs):

        # entities = []
        # scen = cls.scenario_map()[scenario]
        # for agent in scen.agentSprites:
        #     if agent[0] == "H":
        #         entities.append(RLVehicle.init_from_scenario(agent))
        #         kwargs["simple_reward"] = False
        #     elif agent[0] == "S":
        #         entities.append(RLVehicle.init_from_scenario(agent[1:]))
        #         kwargs["simple_reward"] = True
        #     else:
        #         entities.append(KeyboardVehicle.init_from_scenario(agent))

        # # add pedestrian
        # if training:
        #     entities.append(
        #         RandomPedestrian(sim_area[0] / 2, sim_area[1] * (7 / 8), "up")
        #     )
        # else:
        #     entities.append(
        #         KeyboardPedestrian(sim_area[0] / 2, sim_area[1] * (7 / 8), "up")
        #     )

        return (cls(sim_area, cls.scenario_map()[scenario].agentList, **kwargs),)

    @staticmethod
    def scenario_map():
        return {
            0: {
                "name": "RL_r",
                "agentSprites": ["RL_right"],
            },
            1: {
                "name": "RL_l",
                "agentSprites": ["RL_left"],
            },
            2: {
                "name": "RL2_r",
                "agentSprites": ["RL_right", "RL_left"],
            },
            3: {
                "name": "RL2_l",
                "agentSprites": ["RL_left", "RL_right"],
            },
            4: {
                "name": "RL_SRL_r",
                "agentSprites": ["RL_right", "SRL_left"],
            },
            5: {
                "name": "RL_SRL_l",
                "agentSprites": ["RL_left", "SRL_right"],
            },
            6: {
                "name": "RL_H_r",
                "agentSprites": ["RL_right", "H_left"],
            },
            7: {
                "name": "RL_H_l",
                "agentSprites": ["RL_left", "H_right"],
            },
            8: {
                "name": "SRL_r",
                "agentSprites": ["SRL_right"],
            },
            9: {
                "name": "SRL_l",
                "agentSprites": ["SRL_left"],
            },
            10: {
                "name": "SRL2_r",
                "agentSprites": ["SRL_right", "SRL_left"],
            },
            11: {
                "name": "SRL2_l",
                "agentSprites": ["SRL_left", "SRL_right"],
            },
            12: {
                "name": "SRL_RL_r",
                "agentSprites": ["SRL_right", "RL_left"],
            },
            13: {
                "name": "SRL_RL_l",
                "agentSprites": ["SRL_left", "RL_right"],
            },
            14: {
                "name": "SRL_H_r",
                "agentSprites": ["SRL_right", "H_left"],
            },
            15: {
                "name": "SRL_H_l",
                "agentSprites": ["SRL_left", "H_right"],
            },
            16: {
                "name": "H_r",
                "agentSprites": ["H_right"],
            },
            17: {
                "name": "H_l",
                "agentSprites": ["H_left"],
            },
            18: {
                "name": "H2",
                "agentSprites": ["H_right", "H_left"],
            },
        }
