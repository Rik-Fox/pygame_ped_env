import os

from pygame_ped_env.envs import RLCrossingSim
from pygame_ped_env.utils.param_parser import param_parser

from stable_baselines3 import A2C, PPO


def model_init(args=param_parser.parse_args()):

    # if log path not specificed then set to default outside of code folder
    if args.log_path is None:
        wkdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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

    scenarios = [16, 17, 18]

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
    


if __name__ == "__main__":
    model_init(args=param_parser.parse_args())