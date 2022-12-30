# from env import TrafficSim
import time
import os
import numpy as np
from pygame_ped_env.envs import RLCrossingSim
from custom_logging import CustomTrackingCallback
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import stable_baselines3.common.callbacks as clbks


#     EvalCallback,
#     StopTrainingOnMaxEpisodes,
#     StopTrainingOnNoModelImprovement,
#     StopTrainingOnRewardThreshold,
#     EveryNTimesteps,
#     CheckpointCallback,
#     CallbackList,


class Main:
    ### IMPORTANT ###
    # must use correct scenario set otherwise will be updated
    # for rewards recieved by other models actions
    # model_path = "shaped_reward_agent"
    # model_path = "simple_reward_agent"
    simple_agent = False
    # these varibles select which model
    # and it's appropriate scenarios to train
    if not simple_agent:
        # (A) attribute based reward agent
        model_path = "shaped_reward_agent"
        simple_reward = False
        scenarios = [*range(0, 8)]
    else:
        # (B) basic reward agent
        model_path = "simple_reward_agent"
        simple_reward = True
        scenarios = [*range(8, 16)]

    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs",
        model_path,
    )

    os.makedirs(log_path, exist_ok=True)

    window = (1280, 720)
    n_envs = 10
    n_episodes = 1e6

    env = make_vec_env(
        RLCrossingSim,
        n_envs,
        env_kwargs={
            "sim_area": (1280, 720),
            "scenarioList": scenarios,
            "human_controlled_ped": False,  # must be False in training
            "human_controlled_car": False,  # must be False if headless : True
            "headless": True,
            "seed": 4321,
            "basic_model": os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs",
                "simple_reward_agent",
                "init_model",
            ),
            "attr_model": os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs",
                "shaped_reward_agent",
                "init_model",
            ),
            "log_path": log_path,
            "speed_coefficient": 1.0,
            "position_coefficient": 1.0,
            "steering_coefficient": 1.0,
        },
        seed=4321,
        monitor_dir=log_path,
    )
    env.reset()

    # env.modelL.set_env(env)

    eval_env = make_vec_env(
        RLCrossingSim,
        1,
        env_kwargs={
            "sim_area": (1280, 720),
            "scenarioList": scenarios,
            "human_controlled_ped": False,  # must be False in training
            "human_controlled_car": False,  # must be False if headless : True
            "headless": True,
            "seed": 4321,
            "basic_model": os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs",
                "simple_reward_agent",
                "init_model",
            ),
            "attr_model": os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs",
                "shaped_reward_agent",
                "init_model",
            ),
            "log_path": log_path,
            "speed_coefficient": 1.0,
            "position_coefficient": 1.0,
            "steering_coefficient": 1.0,
        },
    )

    callbacks = clbks.CallbackList(
        [
            clbks.EveryNTimesteps(
                1000 * n_envs,
                CustomTrackingCallback(
                    monitor_dir=log_path,
                ),
            ),
            clbks.CheckpointCallback(
                int(1e5),
                os.path.join(log_path, "checkpoints"),
                name_prefix="model_at",
                verbose=1,
            ),
            clbks.StopTrainingOnMaxEpisodes(n_episodes, verbose=1),
            clbks.EvalCallback(
                eval_env=eval_env,
                callback_after_eval=clbks.StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=int(1e5), min_evals=int(1e6), verbose=1
                ),
                callback_on_new_best=clbks.StopTrainingOnRewardThreshold(
                    reward_threshold=(
                        eval_env.envs[0].get_max_reward(simple_reward) / 2
                    )
                    * 0.98
                ),
                n_eval_episodes=5,
                eval_freq=int(1e4),
                log_path=log_path,
                best_model_save_path=os.path.join(log_path, "best"),
                render=False,
                verbose=1,
            ),
        ],
    )

    env.envs[0].modelL.learn(
        total_timesteps=450 * n_episodes,
        tb_log_name="tb_logs",
        callback=callbacks,
    )

    env.modelL.save(os.path.join(log_path, "final_model"), include="env")


if __name__ == "__main__":
    Main()
