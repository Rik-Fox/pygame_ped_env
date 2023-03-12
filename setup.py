from setuptools import setup

setup(
    name="pygame_ped_env",
    version="0.2.8",
    install_requires=[
        "stable-baselines3==1.7.0",
        "sb3-contrib==1.7.0",
        "tensorboard==2.10.0",
        "pygame==2.1.2",
    ],
)
