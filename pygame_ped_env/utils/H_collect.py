import os

from pygame_ped_env.envs import RLCrossingSim
from pygame_ped_env.utils.param_parser import param_parser


def H_collect(args=param_parser.parse_args()):

    # if log path not specificed then set to default outside of code folder
    if args.log_path is None:
        wkdir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
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

    log_path = os.path.join(
        args.log_path,
        args.model_name,
        "H_collect_logs",
    )
    os.makedirs(log_path, exist_ok=True)

    # dont collect for H2 as just load both H_l and H_r
    # [RL_H_r, RL_H_l, SRL_H_r, SLH_H_l, H_r, H_l]
    # scenarios = [6, 7, 14, 15, 16, 17]

    scenarios = [16, 17]

    env = RLCrossingSim(
        sim_area=args.sim_area,
        # scenarioList=args.scenarioList,
        scenarioList=scenarios,
        human_controlled_ped=args.human_controlled_ped,
        # human_controlled_car=args.human_controlled_car,
        human_controlled_car=True,
        # headless=args.headless,
        headless=False,
        seed=args.seed,
        basic_model=args.basic_model,
        attr_model=args.attr_model,
        log_path=log_path,
        speed_coefficient=args.speed_coefficient,
        position_coefficient=args.position_coefficient,
        steering_coefficient=args.steering_coefficient,
    )

    n_episodes = 20

    # for ep in range(args.n_episodes):
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            obs, reward, done, info = env.step({"obs": obs})
        print(info)


if __name__ == "__main__":

    H_collect(args=param_parser.parse_args())
