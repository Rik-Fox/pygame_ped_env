import argparse


def str_2_int(s):
    # import pdb

    # pdb.set_trace()
    return int(s)


# argument parser
param_parser = argparse.ArgumentParser(description="RLCrossingSim")

# agent params
param_parser.add_argument(
    "--shaped_agent",
    default=False,
    action="store_true",
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
param_parser.add_argument("--n_episodes", type=int, default=1000000)
param_parser.add_argument("--n_envs", type=int, default=10)
param_parser.add_argument("--headless", default=True, action="store_false")
param_parser.add_argument("--render", default=False, action="store_true")
param_parser.add_argument("--verbose", type=int, default=1)

# scenario params
param_parser.add_argument("--scenarioList", nargs="+", type=int, default=None)
param_parser.add_argument("--sim_area", nargs=2, type=int, default=[640, 480])
# must be False in training
param_parser.add_argument("--human_controlled_ped", default=False, action="store_true")
# must be False if headless : True
param_parser.add_argument("--human_controlled_car", default=False, action="store_true")

# seed and log params
param_parser.add_argument("--seed", type=int, default=4321)
param_parser.add_argument("--log_interval", type=int, default=1000)
param_parser.add_argument("--checkpoint_interval", type=int, default=1000000)
param_parser.add_argument("--checkpoint_dirname", type=str, default="checkpoints")
param_parser.add_argument("--checkpoint_filename_prefix", type=str, default="model_at")
param_parser.add_argument("--log_path", type=str, default=None)
param_parser.add_argument("--model_name", type=str, default="simple_reward_agent")
param_parser.add_argument("--exp_log_name", type=str, default="train_logs")
# param_parser.add_argument("--monitor_dirname", type=str, default="monitor_files")

# eval params
param_parser.add_argument("--eval_interval", type=int, default=1000000)
param_parser.add_argument("--eval_episodes", type=int, default=10)
param_parser.add_argument("--eval_n_envs", type=int, default=1)
param_parser.add_argument("--eval_headless", default=True, action="store_false")
param_parser.add_argument("--eval_render", default=False, action="store_true")
param_parser.add_argument("--eval_verbose", type=int, default=1)

# eval scenario params
param_parser.add_argument(
    "--eval_human_controlled_ped", default=False, action="store_true"
)
param_parser.add_argument(
    "--eval_human_controlled_car", default=False, action="store_true"
)
param_parser.add_argument("--eval_scenarioList", nargs="+", type=int, default=None)
param_parser.add_argument("--eval_sim_area", nargs=2, type=int, default=[640, 480])
param_parser.add_argument("--eval_speed_coefficient", type=float, default=1.0)
param_parser.add_argument("--eval_position_coefficient", type=float, default=1.0)
param_parser.add_argument("--eval_steering_coefficient", type=float, default=1.0)
param_parser.add_argument("--eval_basic_model", type=str, default=None)
param_parser.add_argument("--eval_attr_model", type=str, default=None)

# eval seed and log params
param_parser.add_argument("--eval_seed", type=int, default=4321)
param_parser.add_argument("--eval_log_name", type=str, default="eval_logs")
# param_parser.add_argument(
#     "--eval_monitor_dirname",
#     type=str,
#     default="monitor_files",
# )

# exploration_initial_eps,
param_parser.add_argument("--exploration_initial_eps", type=float, default=1.0)
param_parser.add_argument("--exploration_fraction", type=float, default=0.1)
param_parser.add_argument("--exploration_final_eps", type=float, default=0.01)
#         exploration_final_eps,
#         exploration_fraction,
