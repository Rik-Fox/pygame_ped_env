# from env import TrafficSim
from pygame_ped_env.agents import (
    RLVehicle,
    KeyboardPedestrian,
    RandomPedestrian,
)  # , Road
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import gym


class Main:

    window = (720, 576)

    agent = RLVehicle([0, window[1] / 2], [window[0], window[1] / 2], "car", "right")

    env = Monitor(
        gym.make(
            "pygame_ped_env:ped_env-v1",
            sim_area=window,
            controllable_sprites=[
                agent,
                # KeyboardPedestrian(window[0] / 2, window[1] * (3 / 4), "up"),
                RandomPedestrian(window[0] / 2, window[1] * (7 / 8), "up"),
            ],
            headless=True,
            seed=1234,
        ),
        "../logs/monitor.csv",
    )

    n_episodes = 1e6
    env.reset()

    # def __init__(
    #     self,
    #     policy: Union[str, Type[DQNPolicy]],
    #     env: Union[GymEnv, str],
    #     learning_rate: Union[float, Schedule] = 1e-4,
    #     buffer_size: int = 1_000_000,  # 1e6
    #     learning_starts: int = 50000,
    #     batch_size: int = 32,
    #     tau: float = 1.0,
    #     gamma: float = 0.99,
    #     train_freq: Union[int, Tuple[int, str]] = 4,
    #     gradient_steps: int = 1,
    #     replay_buffer_class: Optional[ReplayBuffer] = None,
    #     replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
    #     optimize_memory_usage: bool = False,
    #     target_update_interval: int = 10000,
    #     exploration_fraction: float = 0.1,
    #     exploration_initial_eps: float = 1.0,
    #     exploration_final_eps: float = 0.05,
    #     max_grad_norm: float = 10,
    #     tensorboard_log: Optional[str] = None,
    #     create_eval_env: bool = False,
    #     policy_kwargs: Optional[Dict[str, Any]] = None,
    #     verbose: int = 0,
    #     seed: Optional[int] = None,
    #     device: Union[th.device, str] = "auto",
    #     _init_setup_model: bool = True,
    # )

    agent.model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

    # def learn(
    #     self,
    #     total_timesteps: int,
    #     callback: MaybeCallback = None,
    #     log_interval: int = 4,
    #     eval_env: Optional[GymEnv] = None,
    #     eval_freq: int = -1,
    #     n_eval_episodes: int = 5,
    #     tb_log_name: str = "DQN",
    #     eval_log_path: Optional[str] = None,
    #     reset_num_timesteps: bool = True,
    # )

    agent.model.learn(
        total_timesteps=int(250 * n_episodes),
        tb_log_name="DQN_testing",
    )

    # for ep in range(n_episodes):
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         obs, reward, done, info = env.step(agent.model.predict(obs))
    # env.run()

    agent.model.save("../logs/test_model")


if __name__ == "__main__":
    Main()