import os
import gym
import numpy as np

from pygame_ped_env.envs import RLCrossingSim
from pygame_ped_env.utils.param_parser import param_parser
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_masks()


def Eval(args=param_parser.parse_args()):
    # if log path not specificed then set to default outside of code folder
    if args.log_path is None:
        wkdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.log_path = os.path.join(wkdir, "logs")

    if args.basic_model is None:
        # args.basic_model = os.path.join(
        #     args.log_path, "simple_reward_agent", "maskedDQN_init_model"
        # )
        args.basic_model = "/home/rfox/PhD/Term1_22-23_Experiements/pygame_ped_env/maskedDQN_at_134000000_steps.zip"
    if args.eval_basic_model is None:
        args.eval_basic_model = args.basic_model

    if args.attr_model is None:
        # args.attr_model = os.path.join(
        #     args.log_path, "shaped_reward_agent", "maskedDQN_init_model"
        # )
        args.attr_model = "/home/rfox/PhD/Term1_22-23_Experiements/pygame_ped_env/maskedDQN_at_134000000_steps.zip"
    if args.eval_attr_model is None:
        args.eval_attr_model = args.attr_model

    ### IMPORTANT ###
    # must use correct scenario set otherwise will be updated for rewards
    # recieved by other models actions; these varibles switch to the shaped model
    # and all it's appropriate scenarios etc to train with only a boolean flag
    args.shaped_agent = True
    if args.shaped_agent:
        # attribute based reward agent
        args.model_name = "shaped_reward_agent"
        args.scenarioList = [*range(0, 8)]

    log_path = os.path.join(
        args.log_path,
        args.model_name,
        "env_eval_logs",
    )
    os.makedirs(log_path, exist_ok=True)

    scenarios = [0, 1]

    env = RLCrossingSim(
        sim_area=args.sim_area,
        # scenarioList=args.scenarioList,
        scenarioList=scenarios,
        human_controlled_ped=args.human_controlled_ped,
        human_controlled_car=args.human_controlled_car,
        # headless=args.headless,
        headless=False,
        # seed=args.seed,
        seed=257,
        basic_model=args.basic_model,
        attr_model=args.attr_model,
        log_path=log_path,
        speed_coefficient=args.speed_coefficient,
        position_coefficient=args.position_coefficient,
        steering_coefficient=args.steering_coefficient,
    )
    env = ActionMasker(env, mask_fn)
    n_episodes = 10

    # for ep in range(args.n_episodes):
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            if env.scenarioName in ("H2", "H_l", "H_r"):
                obs, reward, done, info = env.step({"obs": obs})
            else:
                obs, reward, done, info = env.step(
                    env.modelL.predict(obs, action_masks=get_action_masks(env))
                )

        print(info)


if __name__ == "__main__":
    Eval(args=param_parser.parse_args())
