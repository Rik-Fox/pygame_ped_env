import os, shutil
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

import ray


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_masks()


@ray.remote(num_cpus=1,num_gpus=1/8,max_retries=-1, retry_exceptions=True)
def Train_variant(speed_coeff, pos_coeff, steer_coeff, args=param_parser.parse_args()):
    args.model_name = (
                    f"{np.round(speed_coeff,2)}_{np.round(pos_coeff,2)}_{np.round(steer_coeff,2)}"
                )
    args.exp_log_name = f"maskedDQN_retrain_logs"
    args.var_log_path = os.path.join(
        args.log_path,
        args.model_name,
        args.exp_log_name,
    )

    args.monitor_path = os.path.join(args.var_log_path, "monitor_files")

    args.var_eval_log_path = os.path.join(
        args.log_path,
        args.model_name,
        args.eval_log_name,
    )

    args.eval_monitor_path = os.path.join(args.var_eval_log_path, "monitor_files")

    os.makedirs(args.var_log_path, exist_ok=True)
    os.makedirs(args.var_eval_log_path, exist_ok=True)
    os.makedirs(args.monitor_path, exist_ok=True)
    os.makedirs(args.eval_monitor_path, exist_ok=True)
    
    cp_path = os.path.join(args.var_log_path,"checkpoints")
    # print("cp_path ",cp_path)  
    os.makedirs(cp_path, exist_ok=True)
    cps = os.listdir(cp_path)
    # print("cps ",cps)
    if cps:
        pass
    else:
        shutil.copy(args.attr_model, os.path.join(cp_path, "maskedDQN_at_0_steps.zip"))
        cps = ["maskedDQN_at_0_steps.zip"]
        
    cp_step_counts = [int(cp.split("_")[2]) for cp in cps]
    # print("cp_step_counts ",cp_step_counts)
    # print("max ",max(cp_step_counts))
    # print("index",cp_step_counts.index(max(cp_step_counts)))
    max_steps = max(cp_step_counts)
    most_recent_cp = cps[cp_step_counts.index(max_steps)]
    
    args.seed = max_steps + np.random.randint(0, 500)
    
    args.attr_model = os.path.join(cp_path, most_recent_cp)
    args.eval_attr_model = os.path.join(cp_path, most_recent_cp)
    
    print(f"starting {np.round(speed_coeff,2)}_{np.round(pos_coeff,2)}_{np.round(steer_coeff,2)} model from checkpoint =>", most_recent_cp)
    

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
            "log_path": args.var_log_path,
            "speed_coefficient": speed_coeff,
            "position_coefficient": pos_coeff,
            "steering_coefficient": steer_coeff,
        },
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        seed=args.seed,
        monitor_dir=args.monitor_path,
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
            "log_path": args.var_eval_log_path,
            "speed_coefficient": speed_coeff,
            "position_coefficient": pos_coeff,
            "steering_coefficient": steer_coeff,
        },
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        seed=args.eval_seed,
        monitor_dir=args.eval_monitor_path,
    )

    model_name = args.exp_log_name.split(os.sep)[-1].split("_")[0]

    callbacks = clbks.CallbackList(
        [
            clbks.EveryNTimesteps(
                args.log_interval,
                CustomTrackingCallback(
                    monitor_dir=args.monitor_path,
                ),
            ),
            clbks.CheckpointCallback(
                args.checkpoint_interval,
                os.path.join(args.var_log_path, args.checkpoint_dirname),
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
                callback_on_new_best=None,
                n_eval_episodes=args.eval_episodes,
                eval_freq=args.eval_interval,
                log_path=args.var_log_path,
                best_model_save_path=os.path.join(args.var_eval_log_path, f"{model_name}_best"),
                render=args.eval_render,
                verbose=args.eval_verbose,
            ),
        ],
    )
    # for i in range(args.n_envs):
    #     env.envs[i].modelL.set_env(env.envs[i])

    loaded_model = env.envs[0].modelA

    env.envs[0].modelL.verbose = args.verbose

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
            ),
            0,
            1,
        ),
    )

    env.envs[0].modelL.learn(
        total_timesteps=1000 * args.n_episodes,
        tb_log_name=os.path.join(args.var_log_path, "tb_logs"),
        callback=callbacks,
        log_interval=args.log_interval / 10,
        reset_num_timesteps=False,
    )

    env.envs[0].modelL.save(os.path.join(args.var_log_path, f"{args.model_name}_final"), include="env")

    return env.envs[0].modelL


def Main(args=param_parser.parse_args()):
    
    

    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)

    output_ids = []        

    coeff_map = {
        0: float(1 / 4),
        1: float(1 / 3),
        2: float(1 / 2),
        3: float(1),
        4: float(2),
        5: float(3),
        6: float(4),
    }

    for k in range(1):
        for j in range(1):
            for i in range(7):
                output_ids.append(
                    Train_variant.remote(coeff_map[i], coeff_map[j], coeff_map[k], args)
                    # Train_variant(coeff_map[i], coeff_map[j], coeff_map[k], args)
                )
    # ray remote is non terminating so need to wait for all to finish
    # .get() is a blocking call making this script wait for all to finish
    for model in ray.get(output_ids):
        print(model._episode_num)


if __name__ == "__main__":

    args = param_parser.parse_args()

    args.shaped_agent = True

    args.n_episodes = int(1e5)

    # if log path not specificed then set to default outside of code folder
    if args.log_path is None:
        wkdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.log_path = os.path.join(wkdir, "logs", "shaped_reward_agent_variants")

    if args.basic_model is None:
        args.basic_model = os.path.join(
            os.path.dirname(args.log_path),
            "simple_reward_agent",
            "maskedDQN_init_model_480",
        )
    if args.eval_basic_model is None:
        args.eval_basic_model = args.basic_model

    if args.attr_model is None:
        print("No model given, using default untrained model!")
        args.attr_model = os.path.join(
            os.path.dirname(args.log_path),
            "shaped_reward_agent",
            "maskedDQN_init_model_480.zip",
        )
        # /home/rawsys/mathgw/term1_22_experiments/logs/shaped_reward_agent/maskedDQN_init_model_480.zip
        # args.attr_model = "/home/rfox/PhD/Term1_22-23_Experiements/pygame_ped_env/maskedDQN_at_172000000_steps.zip"

    if args.eval_attr_model is None:
        args.eval_attr_model = args.attr_model

    if args.scenarioList is None:
        args.scenarioList = [0, 1]
        args.eval_scenarioList = [0, 1]

    args.verbose = 0

    args.exploration_initial_eps = 0.5
    args.exploration_final_eps = 0.01
    args.exploration_fraction = 0.25

    Main(args=args)
