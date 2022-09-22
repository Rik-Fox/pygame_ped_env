# from env import TrafficSim
import time
import os
from pygame_ped_env.agents import (
    RLVehicle,
    KeyboardPedestrian,
    RandomPedestrian,
)  # , Road
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import gym

from custom_logging import CustomTrackingCallback


class Eval:

    # log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs")
    # print("log_path => ",str(log_path))

    model_save_path = "/home/rfox/PhD/Term1_22-23_Experiements/logs/DQN_testing_10"

    window = (720, 576)

    agent = RLVehicle([0, window[1] / 2], [window[0], window[1] / 2], "car", "right")

    agent.model = DQN.load(os.path.join(model_save_path, "worst"))

    env = agent.model.get_env()
    if env is None:
        agent.model.set_env(
            gym.make(
                "pygame_ped_env:ped_env-v1",
                sim_area=window,
                controllable_sprites=[
                    agent,
                    # KeyboardPedestrian(window[0] / 2, window[1] * (3 / 4), "up"),
                    RandomPedestrian(window[0] / 2, window[1] * (7 / 8), "up"),
                ],
                headless=False,
                seed=4321,
            )
        )
        env = agent.model.get_env()

    n_episodes = 6

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            obs, reward, done, info = env.step(agent.model.predict(obs))


if __name__ == "__main__":
    Eval()
