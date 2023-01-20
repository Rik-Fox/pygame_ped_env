import os
import gym
import numpy as np

from pygame_ped_env.envs import RLCrossingSim
from pygame_ped_env.utils.param_parser import param_parser
from pygame_ped_env.entities.maskedDQN import MaskableDQNPolicy, MaskableDQN

from stable_baselines3 import A2C, PPO, DQN


from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()


def model_init(args=param_parser.parse_args()):

    # if log path not specificed then set to default outside of code folder
    print(args.log_path)
    if args.log_path is None:
        wkdir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        args.log_path = os.path.join(wkdir, "logs")

    print(args.log_path)

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
    if args.shaped_agent:
        # attribute based reward agent
        args.model_name = "shaped_reward_agent"
        args.scenarioList = [*range(0, 8)]

    log_path = os.path.join(
        args.log_path,
        args.model_name,
        "env_eval_logs",
    )
    print(log_path)
    os.makedirs(log_path, exist_ok=True)

    scenarios = [0, 1]

    env = RLCrossingSim(
        sim_area=args.sim_area,
        # scenarioList=args.scenarioList,
        scenarioList=scenarios,
        human_controlled_ped=args.human_controlled_ped,
        human_controlled_car=args.human_controlled_car,
        headless=args.headless,
        seed=args.seed,
        basic_model=args.basic_model,
        attr_model=args.attr_model,
        log_path=log_path,
        speed_coefficient=args.speed_coefficient,
        position_coefficient=args.position_coefficient,
        steering_coefficient=args.steering_coefficient,
    )

    model = A2C("MlpPolicy", env, verbose=1)
    model.save(os.path.join(log_path, "a2c_init_model"))

    model = PPO("MlpPolicy", env, verbose=1)
    model.save(os.path.join(log_path, "ppo_init_model"))

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model.save(os.path.join(log_path, "masked_ppo_init_model"))

    model = MaskableDQN(MaskableDQNPolicy, env, verbose=1)
    model.save(os.path.join(log_path, "masked_dqn_init_model"))


if __name__ == "__main__":
    model_init(args=param_parser.parse_args())
