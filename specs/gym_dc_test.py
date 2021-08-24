"""A simple demo that produces an image from the environment."""
import os
from ml_logger import logger
import numpy as np

import gym
import distracting_control

if __name__ == '__main__':
    org_ml_logger_root = os.environ.get('ML_LOGGER_ROOT')
    os.environ['ML_LOGGER_ROOT'] = ''

    for i, difficulty in enumerate(['easy', 'medium', 'hard']):
        for domain, task in [
            ('ball_in_cup', 'catch'),
            # ('cartpole', 'swingup'),
            # ('cheetah', 'run'),
            # ('finger', 'spin'),
            # ('reacher', 'easy'),
            # ('walker', 'walk')
        ]:
            # NOTE:
            # f'{domain.capitalize()}-{task}-{difficulty}-v1' loads a distracting env
            # f'{domain.capitalize()}-{task}-v1' loads a non-distracting env
            env = gym.make(f'{domain.capitalize()}-{task}-{difficulty}-v1', distracting_seed=0,
                           from_pixels=True, channels_first=False, dynamic=False, fix_distraction=True)

            prefix = f"reprod_figures/fix_distraction/{domain}-{task}-{difficulty}"
            n_trials = 5
            episode_length = 10
            for j in range(n_trials):
                obs = env.reset()
                done = False
                counter = 0
                for k in range(episode_length):
                    print(i, j, k)
                    logger.save_image(obs, os.path.join(prefix, f"ep{j}-{k}.png"))
                    act = env.action_space.sample()
                    obs, reward, done, info = env.step(act)
                    if done:
                        break


            mock_env = gym.make(f'{domain.capitalize()}-{task}-{difficulty}-v1', distracting_seed=1,
                           from_pixels=True, channels_first=False, dynamic=False, fix_distraction=True)
            obs = mock_env.reset()

            prefix = f"reprod_figures/save_and_load/{domain}-{task}-{difficulty}"
            logger.save_image(obs, os.path.join(prefix, "original_obs.png"))

            _state = mock_env.get_distracting_state()

            import torch
            os.makedirs(prefix, exist_ok=True)
            with open(os.path.join(prefix, 'pickled_state.pkl'), 'wb') as f:
                torch.save(_state, f)
            print('loading from pickle..')
            with open(os.path.join(prefix, 'pickled_state.pkl'), 'rb') as f:
                state = torch.load(f)

            # logger.save_torch(_state, os.path.join(prefix, "pickled_state.pkl"))
            # state = logger.load_torch(os.path.join(prefix, "pickled_state.pkl"))

            n_trials = 5
            for j in range(n_trials):
                # Load distractions from the pickled state
                # NOTE: distracting_seed should be disregarded
                env = gym.make(f'{domain.capitalize()}-{task}-{difficulty}-v1', distracting_seed=j * 100,
                            from_pixels=True, channels_first=False, dynamic=False, fix_distraction=True,
                            distraction_dict=state)
                obs = env.reset()
                logger.save_image(obs, os.path.join(prefix, f"{j}.png"))

    os.environ['ML_LOGGER_ROOT'] = org_ml_logger_root
