from gym.envs.registration import register

register(
    id="ped_env-v0",
    entry_point="pygame_ped_env.envs:TrafficSim",
)
register(
    id="ped_env-v1",
    entry_point="pygame_ped_env.envs:RLCrossingSim",
)
