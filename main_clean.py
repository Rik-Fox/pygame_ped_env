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


class Main:

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", time.asctime())
    print("log_path => ",str(log_path))

    window = (720, 576)

    agent = RLVehicle([0, window[1] / 2], [window[0], window[1] / 2], "car", "right")

    env = make_vec_env(
        "pygame_ped_env:ped_env-v1",
        10,
        env_kwargs={
            "sim_area": window,
            "controllable_sprites": [
                agent,
                RandomPedestrian(window[0] / 2, window[1] * (7 / 8), "up"),
            ],
            "headless": True,
        },
        seed=4321,
        monitor_dir=log_path,
    )

    n_episodes = 1e6
    env.reset()

    agent.model = DQN("MlpPolicy", env, verbose=0, tensorboard_log=log_path)


    agent.model.learn(
        total_timesteps= 250 * n_episodes,
        tb_log_name="DQN_testing",
        callback=CustomTrackingCallback(
            check_freq=1000,
            monitor_dir=log_path,
            start_time=time.time(),
            verbose=1,
        ),
    )

    agent.model.save(os.path.join(log_path,"test_model"), include="env")


if __name__ == "__main__":
    Main()
