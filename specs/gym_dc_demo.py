"""A simple demo that produces an image from the environment."""
import gym
import os
from ml_logger import logger

import distracting_control.suite as suite

if __name__ == '__main__':
    suite.BG_DATA_PATH = f"{os.environ['HOME']}/datasets/DAVIS/JPEGImages/480p"

    for i, difficulty in enumerate(['easy', 'medium', 'hard']):
        for domain, task in [
            ('ball_in_cup', 'catch'),
            ('cartpole', 'swingup'),
            ('cheetah', 'run'),
            ('finger', 'spin'),
            ('reacher', 'easy'),
            ('walker', 'walk')
        ]:
            env = gym.make(f'distracting_control:{domain.capitalize()}-{task}-{difficulty}-v1', from_pixels=True)

            obs = env.reset()
            act = env.action_space.sample()
            obs, reward, done, info = env.step(act)

            logger.save_image(obs, f"figures/{domain}-{task}-({difficulty}).png")
