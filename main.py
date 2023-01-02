# from env import TrafficSim
import time
import os
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import stable_baselines3.common.callbacks as clbks

from pygame_ped_env.envs import RLCrossingSim
from custom_logging import CustomTrackingCallback
from param_parser import *

#     EvalCallback,
#     StopTrainingOnMaxEpisodes,
#     StopTrainingOnNoModelImprovement,
#     StopTrainingOnRewardThreshold,
#     EveryNTimesteps,
#     CheckpointCallback,
#     CallbackList,


def Main(args=param_parser.parse_args()):

    # if log path not specificed then set to default outside of code folder
    if args.log_path is None:
        wkdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.log_path = os.path.join(wkdir, "logs")

    if args.basic_model is None:
        args.basic_model = os.path.join(
            args.log_path, "simple_reward_agent", "init_model"
        )

    if args.eval_basic_model is None:
        args.eval_basic_model = args.basic_model

    if args.attr_model is None:
        args.attr_model = os.path.join(
            args.log_path, "shaped_reward_agent", "init_model"
        )

    if args.eval_attr_model is None:
        args.eval_attr_model = args.attr_model

    ### IMPORTANT ###
    # must use correct scenario set otherwise will be updated for rewards
    # recieved by other models actions; these varibles switch to the shaped model
    # and all it's appropriate scenarios etc to train with only a boolean flag
    if not args.simple_agent:
        # attribute based reward agent
        args.model_name = "shaped_reward_agent"
        args.scenarioList = [*range(0, 8)]
        args.eval_scenarioList = [*range(0, 8)]
        args.monitor_path = os.path.join(
            args.log_path, args.model_name, args.exp_log_name, "monitor"
        )
        args.eval_monitor_path = os.path.join(
            args.log_path, args.model_name, args.eval_log_name, "eval_monitor"
        )

    log_path = os.path.join(
        args.log_path,
        args.model_name,
        args.exp_log_name,
    )

    eval_log_path = os.path.join(
        args.log_path,
        args.model_name,
        args.eval_log_name,
    )

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(eval_log_path, exist_ok=True)

    n_envs = 10
    n_episodes = 1e6

    env = make_vec_env(
        RLCrossingSim,
        args.n_envs,
        env_kwargs={
            "sim_area": args.sim_area,
            "scenarioList": args.scenarioList,
            "human_controlled_ped": args.human_controlled_ped,
            "human_controlled_car": args.human_controlled_car,
            "headless": args.headless,
            "seed": args.seed,
            "basic_model": args.basic_model,
            "attr_model": args.attr_model,
            "log_path": log_path,
            "speed_coefficient": args.speed_coefficient,
            "position_coefficient": args.position_coefficient,
            "steering_coefficient": args.steering_coefficient,
        },
        seed=args.seed,
        monitor_dir=args.monitor_path,
    )
    env.reset()

    # TODO: add seperate eval env params in param_parser

    eval_env = make_vec_env(
        RLCrossingSim,
        args.eval_n_envs,
        env_kwargs={
            "sim_area": args.eval_sim_area,
            "scenarioList": args.eval_scenarioList,
            "human_controlled_ped": args.eval_human_controlled_ped,
            "human_controlled_car": args.eval_human_controlled_car,
            "headless": args.eval_headless,
            "seed": args.eval_seed,
            "basic_model": args.eval_basic_model,
            "attr_model": args.eval_attr_model,
            "log_path": eval_log_path,
            "speed_coefficient": args.eval_speed_coefficient,
            "position_coefficient": args.eval_position_coefficient,
            "steering_coefficient": args.eval_steering_coefficient,
        },
        seed=args.eval_seed,
        monitor_dir=args.eval_monitor_path,
    )

    callbacks = clbks.CallbackList(
        [
            clbks.EveryNTimesteps(
                args.log_interval * args.n_envs,
                CustomTrackingCallback(
                    monitor_dir=args.monitor_path,
                ),
            ),
            clbks.CheckpointCallback(
                args.checkpoint_interval * args.n_envs,
                os.path.join(log_path, args.checkpoint_dirname),
                name_prefix=args.checkpoint_filename_prefix,
                verbose=args.verbose,
            ),
            clbks.StopTrainingOnMaxEpisodes(n_episodes, verbose=args.verbose),
            clbks.EvalCallback(
                eval_env=eval_env,
                callback_after_eval=clbks.StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=int(1e5),
                    min_evals=int(1e6),
                    verbose=args.eval_verbose,
                ),
                callback_on_new_best=clbks.StopTrainingOnRewardThreshold(
                    reward_threshold=(
                        eval_env.envs[0].get_max_reward(args.simple_agent) / 2
                    )
                    * 0.98
                ),
                n_eval_episodes=args.eval_episodes,
                eval_freq=args.eval_interval * args.eval_n_envs,
                log_path=log_path,
                best_model_save_path=os.path.join(eval_log_path, "best"),
                render=args.eval_render,
                verbose=args.eval_verbose,
            ),
        ],
    )

    env.envs[0].modelL.learn(
        total_timesteps=450 * n_episodes,
        tb_log_name=os.path.join(log_path, "tb_logs"),
        callback=callbacks,
    )

    env.modelL.save(os.path.join(log_path, "final_model"), include="env")


if __name__ == "__main__":

    args = param_parser.parse_args()

    Main(args)
