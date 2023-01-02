import argparse
import os

# argument parser
param_parser = argparse.ArgumentParser(description="RLCrossingSim")

# agent params
param_parser.add_argument(
    "--simple_agent",
    type=bool,
    default=True,
    help="if True, train simple reward agent, else train shaped reward agent",
)
param_parser.add_argument(
    "--basic_model",
    type=str,
    default=None,
)
param_parser.add_argument(
    "--attr_model",
    type=str,
    default=None,
)

# reward params
param_parser.add_argument("--speed_coefficient", type=float, default=1.0)
param_parser.add_argument("--position_coefficient", type=float, default=1.0)
param_parser.add_argument("--steering_coefficient", type=float, default=1.0)

# training params
param_parser.add_argument("--n_episodes", type=int, default=10)
param_parser.add_argument("--n_envs", type=int, default=10)
param_parser.add_argument("--headless", type=bool, default=True)
param_parser.add_argument("--render", type=bool, default=False)
param_parser.add_argument("--verbose", type=int, default=1)

# scenario params
param_parser.add_argument("--scenarioList", type=list, default=[*range(8, 16)])
param_parser.add_argument("--sim_area", type=tuple, default=(1280, 720))
# must be False in training
param_parser.add_argument("--human_controlled_ped", type=bool, default=False)
# must be False if headless : True
param_parser.add_argument("--human_controlled_car", type=bool, default=False)

# seed and log params
param_parser.add_argument("--seed", type=int, default=4321)
param_parser.add_argument("--log_interval", type=int, default=1000)
param_parser.add_argument("--checkpoint_interval", type=int, default=1000)
param_parser.add_argument("--checkpoint_dirname", type=str, default="checkpoints")
param_parser.add_argument("--checkpoint_filename_prefix", type=str, default="model_at")
param_parser.add_argument("--log_path", type=str, default=None)
param_parser.add_argument("--model_name", type=str, default="simple_reward_agent")
param_parser.add_argument("--exp_log_name", type=str, default="train_logs")
param_parser.add_argument(
    "--monitor_path", type=str, default="logs/simple_reward_agent/train_logs/monitor"
)

# eval params
param_parser.add_argument("--eval_interval", type=int, default=1000)
param_parser.add_argument("--eval_episodes", type=int, default=10)
param_parser.add_argument("--eval_n_envs", type=int, default=1)
param_parser.add_argument("--eval_headless", type=bool, default=True)
param_parser.add_argument("--eval_render", type=bool, default=False)
param_parser.add_argument("--eval_verbose", type=int, default=1)

# eval scenario params
param_parser.add_argument("--eval_human_controlled_ped", type=bool, default=False)
param_parser.add_argument("--eval_human_controlled_car", type=bool, default=False)
param_parser.add_argument("--eval_seed", type=int, default=4321)
param_parser.add_argument("--eval_scenarioList", type=list, default=[*range(8, 16)])
param_parser.add_argument("--eval_sim_area", type=tuple, default=(1280, 720))
param_parser.add_argument("--eval_speed_coefficient", type=float, default=1.0)
param_parser.add_argument("--eval_position_coefficient", type=float, default=1.0)
param_parser.add_argument("--eval_steering_coefficient", type=float, default=1.0)
param_parser.add_argument("--eval_basic_model", type=str, default=None)
param_parser.add_argument("--eval_attr_model", type=str, default=None)

# eval seed and log params
param_parser.add_argument("--eval_log_name", type=str, default="eval_logs")
param_parser.add_argument(
    "--eval_monitor_path",
    type=str,
    default="logs/simple_reward_agent/eval_logs/monitor",
)
