import os
import gym
import numpy as np

from stable_baselines3.common import utils
from stable_baselines3.common.env_util import make_vec_env
import stable_baselines3.common.callbacks as clbks
from sb3_contrib.common.wrappers import ActionMasker

#     EvalCallback,
#     StopTrainingOnMaxEpisodes,
#     StopTrainingOnNoModelImprovement,
#     StopTrainingOnRewardThreshold,
#     EveryNTimesteps,
#     CheckpointCallback,
#     CallbackList,

from pygame_ped_env.envs import RLCrossingSim
from pygame_ped_env.utils.param_parser import param_parser
from pygame_ped_env.utils.custom_logging import CustomTrackingCallback


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_masks()


def Main(args=param_parser.parse_args()):

    ### IMPORTANT ###
    # must use correct scenario set otherwise will be updated for rewards
    # recieved by other models actions; these varibles switch to the shaped model
    # and all it's appropriate scenarios etc to train with only a boolean flag
    if args.shaped_agent:
        # attribute based reward agent
        args.model_name = "shaped_reward_agent"

    log_path = os.path.join(
        args.log_path,
        args.model_name,
        args.exp_log_name,
    )

    monitor_path = os.path.join(log_path, "monitor_files")

    eval_log_path = os.path.join(
        args.log_path,
        args.model_name,
        args.eval_log_name,
    )

    eval_monitor_path = os.path.join(eval_log_path, "monitor_files")

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(eval_log_path, exist_ok=True)
    os.makedirs(monitor_path, exist_ok=True)
    os.makedirs(eval_monitor_path, exist_ok=True)

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
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        seed=args.seed,
        monitor_dir=monitor_path,
    )
    # env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    env.reset()

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
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        seed=args.eval_seed,
        monitor_dir=eval_monitor_path,
    )

    model_name = args.exp_log_name.split(os.sep)[-1].split("_")[0]

    callbacks = clbks.CallbackList(
        [
            clbks.EveryNTimesteps(
                args.log_interval,
                CustomTrackingCallback(
                    monitor_dir=monitor_path,
                ),
            ),
            clbks.CheckpointCallback(
                args.checkpoint_interval,
                os.path.join(log_path, args.checkpoint_dirname),
                name_prefix=f"{model_name}_at",
                verbose=args.verbose,
            ),
            clbks.StopTrainingOnMaxEpisodes(args.n_episodes, verbose=args.verbose),
            clbks.EvalCallback(
                eval_env=eval_env,
                callback_after_eval=clbks.StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=int(1e5),
                    min_evals=int(1e6),
                    verbose=args.eval_verbose,
                ),
                callback_on_new_best=clbks.StopTrainingOnRewardThreshold(
                    reward_threshold=0
                    # reward_threshold=(
                    #     eval_env.envs[0].get_max_reward(args.shaped_agent) / 2
                    # )
                    # * 0.98
                ),
                n_eval_episodes=args.eval_episodes,
                eval_freq=args.eval_interval,
                log_path=log_path,
                best_model_save_path=os.path.join(eval_log_path, f"{model_name}_best"),
                render=args.eval_render,
                verbose=args.eval_verbose,
            ),
        ],
    )
    # for i in range(args.n_envs):
    #     env.envs[i].modelL.set_env(env.envs[i])

    env.envs[0].modelL.verbose = args.verbose

    if args.cont:
        if args.shaped_agent:
            loaded_model = env.envs[0].modelA
        else:
            loaded_model = env.envs[0].modelB

        env.envs[0].modelL._episode_num = loaded_model._episode_num
        env.envs[0].modelL._n_calls = loaded_model._n_calls
        env.envs[0].modelL._n_updates = loaded_model._n_updates
        env.envs[0].modelL.num_timesteps = loaded_model.num_timesteps
        env.envs[0].modelL.replay_buffer = loaded_model.replay_buffer
        env.envs[0].modelL.learning_starts = 0
        # env.envs[0].modelL.load_replay_buffer(loaded_model.replay_buffer)
        env.envs[0].modelL.exploration_schedule = utils.get_linear_fn(
            loaded_model.exploration_rate,
            loaded_model.exploration_final_eps,
            np.clip(
                (
                    loaded_model._current_progress_remaining
                    - (1 - loaded_model.exploration_fraction)
                )
                / loaded_model.exploration_fraction,
                0,
                1,
            ),
        )
        total_timesteps = loaded_model._total_timesteps - loaded_model.num_timesteps
        # tb_log_name = loaded_model.tensorboard_log
        # log_interval = loaded_model.log_interval
        reset_num_timesteps = False
    else:
        env.envs[0].modelL.exploration_schedule = utils.get_linear_fn(
            args.exploration_initial_eps,
            args.exploration_final_eps,
            args.exploration_fraction,
        )
        total_timesteps = 1000 * args.n_episodes
        # tb_log_name = os.path.join(log_path, "tb_logs")
        # log_interval = args.log_interval / 10
        reset_num_timesteps = True

    env.envs[0].modelL.learn(
        total_timesteps=total_timesteps,
        tb_log_name=os.path.join(log_path, "tb_logs"),
        callback=callbacks,
        log_interval=args.log_interval / 10,
        reset_num_timesteps=reset_num_timesteps,
        # log_interval=10,
    )

    env.envs[0].modelL.save(os.path.join(log_path, "final_model"), include="env")


if __name__ == "__main__":

    args = param_parser.parse_args()

    # if log path not specificed then set to default outside of code folder
    if args.log_path is None:
        wkdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.log_path = os.path.join(wkdir, "logs")

    if args.basic_model is None:
        # args.basic_model = os.path.join(
        #     args.log_path, "simple_reward_agent", "maskedDQN_init_model_480"
        # )
        args.basic_model = os.path.join(
            args.log_path, "simple_reward_agent", "maskedDQN_init_model_480"
        )

    if args.eval_basic_model is None:
        args.eval_basic_model = args.basic_model

    if args.attr_model is None:
        args.attr_model = os.path.join(
            args.log_path, "shaped_reward_agent", "maskedDQN_init_model_480"
        )
        # args.attr_model = "/home/rfox/PhD/Term1_22-23_Experiements/logs/shaped_reward_agent/maskedDQN_train_neg/checkpoints/maskedDQN_at_48000000_steps"

    if args.eval_attr_model is None:
        args.eval_attr_model = args.attr_model

    if args.scenarioList is None:
        if args.shaped_agent:
            args.scenarioList = [0, 1]
            args.eval_scenarioList = [0, 1]
        else:
            args.scenarioList = [8, 9]
            args.eval_scenarioList = [8, 9]
    else:
        args.scenarioList = [int(i) for i in args.scenarioList]
        args.eval_scenarioList = args.scenarioList

    Main(args=args)
