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

    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs", "eval_logs", time.asctime()
    )
    # print("log_path => ",str(log_path))

    window = (720, 576)

    agent = RLVehicle(window, "car", "right")
    agentL = RLVehicle(window, "car", "left")

    model_save_path = None

    if model_save_path:
        agent.model = DQN.load(os.path.join(model_save_path, "best"))

    try:
        env = agent.model.get_env()
        if env is None:
            agent.model.set_env(
                gym.make(
                    "pygame_ped_env:ped_env-v1",
                    sim_area=window,
                    controllable_sprites=[
                        agent,
                        agentL,
                        # KeyboardPedestrian(window[0] / 2, window[1] * (3 / 4), "up"),
                        RandomPedestrian(window[0] / 2, window[1] * (7 / 8), "up"),
                    ],
                    headless=False,
                    simple_reward=False,
                    seed=4321,
                )
            )
            env = agent.model.get_env()

    except AttributeError:
        # env = gym.make(
        #     "pygame_ped_env:ped_env-v1",
        #     sim_area=window,
        #     controllable_sprites=[
        #         agent,
        #         agentL,
        #         # KeyboardPedestrian(window[0] / 2, window[1] * (3 / 4), "up"),
        #         RandomPedestrian(window[0] / 2, window[1] * (7 / 8), "up"),
        #     ],
        #     headless=False,
        #     simple_reward=False,
        #     seed=4321,
        # )
        env = RLCrossingSim(
            window,
            # [
            #     agent,
            #     agentL,
            #     RandomPedestrian(window[0] / 2, window[1] * (7 / 8), "up"),
            # ],
            headless=False,
            seed=4321,
        )
        agent.model = DQN("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
        agentL.model = DQN("MlpPolicy", env, verbose=0, tensorboard_log=log_path)

    n_episodes = 6

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            obs, reward, done, info = env.step(agent.model.predict(obs))


if __name__ == "__main__":
    Eval()
