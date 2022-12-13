# from env import TrafficSim
import time
import os
from pygame_ped_env.agents import (
    RLVehicle,
    KeyboardPedestrian,
    RandomPedestrian,
)  # , Road
from pygame_ped_env.envs import RLCrossingSim
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import gym

from custom_logging import CustomTrackingCallback


class Eval:

    # (A) attribute based reward agent
    attr_save_path = "shaped_reward_agent"
    train_setA = [*range(0, 8)]

    # (B) basic reward agent
    basic_save_path = "simple_reward_agent"
    train_setB = [*range(8, 16)]

    # these varibles select which model
    # and it's appropriate scenarios to train
    model_path = attr_save_path
    scenarios = train_setA

    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs",
        "eval_logs",
        model_path,
        # time.asctime(),
    )

    env = RLCrossingSim(
        window=(1280, 720),
        scenarioList=scenarios,
        human_controlled_ped=False,
        human_controlled_car=False,
        headless=False,
        seed=4321,
        simple_model=basic_save_path,
        attr_model=attr_save_path,
        log_path=log_path,
        speed_coefficient=1.0,
        position_coefficient=1.0,
        steering_coefficient=1.0,
    )

    n_episodes = 10

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            obs, reward, done, info = env.step(env.modelL.predict(obs))


if __name__ == "__main__":
    Eval()
