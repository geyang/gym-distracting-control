"""A simple demo that produces an image from the environment."""
import os
from ml_logger import logger
import numpy as np

import gym
import distracting_control
import time

class Time:
    accm_saveimg = 0
    accm_step = 0
    accm_reset = 0

from contextlib import contextmanager

@contextmanager
def mllogger_root_to(value=''):
    org_ml_logger_root = os.environ.get('ML_LOGGER_ROOT', "")
    os.environ['ML_LOGGER_ROOT'] = value
    try:
        yield
    finally:
        os.environ['ML_LOGGER_ROOT'] = org_ml_logger_root


def compile_images(outdir='reprod_figures/intensities'):
    with mllogger_root_to():
        intensities = np.linspace(0, 1, 10 + 1)
        for distraction in ['color', 'camera']:
            for domain, task in [
                ('ball_in_cup', 'catch'),
                ('cartpole', 'swingup'),
                ('cheetah', 'run'),
                ('finger', 'spin'),
                ('reacher', 'easy'),
                ('walker', 'walk')
            ]:
                prefix = f"{outdir}/{distraction}/{domain}-{task}"
                observations = []
                for i, intensity in enumerate(intensities):
                    print(distraction, prefix, intensity)

                    env = gym.make(
                        f'distracting_control:{domain.capitalize()}-{task}-intensity-v1',
                        from_pixels=True, channels_first=False, dynamic=False, fix_distraction=True,
                        intensity=intensity, distraction_types=(distraction,), sample_from_edge=True
                    )
                    env.reset()
                    observations.append(env.unwrapped.env.physics.render(height=128, width=128, camera_id=0))

                logger.save_image(np.hstack(observations), prefix + "-intensities.png")


def record_video(outdir='reprod_figures/intensities/video'):
    with mllogger_root_to():
        domain, task = 'cheetah', 'run'
        intensity = 0.3
        for distraction in ['color', 'camera']:
            for distr_seed in [val * 100 for val in range(5)]:
                prefix = f"{outdir}/{distraction}/{domain}-{task}-intensity{intensity}-seed{distr_seed}"
                n_trials = 5
                episode_length = 10
                for j in range(n_trials):
                    print(prefix, j)

                    env = gym.make(
                        f'distracting_control:{domain.capitalize()}-{task}-intensity-v1',
                        from_pixels=True, channels_first=False, dynamic=False, fix_distraction=True,
                        intensity=intensity, distraction_types=(distraction, ), sample_from_edge=True,
                        distraction_seed=distr_seed
                    )
                    obs = env.reset()

                    done = False
                    frames = []
                    for k in range(episode_length):
                        frames.append(env.unwrapped.env.physics.render(height=256, width=256, camera_id=0))
                        act = env.action_space.sample()
                        obs, reward, done, info = env.step(act)
                        if done:
                            break

                    logger.save_video(frames, prefix + f'ep{j}.mp4')


if __name__ == '__main__':
    print("---------- genrating images ----------")
    compile_images()
    print("---------- genrating videos ----------")
    record_video()
