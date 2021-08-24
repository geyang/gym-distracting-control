from dm_control.suite import ALL_TASKS
from gym.envs import register


def make_env(flatten_obs=True, from_pixels=False, frame_skip=1, max_episode_steps=1000, distracting_seed=0, **kwargs):
    max_episode_steps /= frame_skip

    from distracting_control.gym_env import DistractingEnv
    seed_kwargs = {key: {'seed': distracting_seed} for key in ('background_kwargs', 'camera_kwargs', 'color_kwargs')}
    env = DistractingEnv(
        from_pixels=from_pixels, frame_skip=frame_skip, **seed_kwargs, **kwargs)
    if from_pixels:
        from .wrappers import ObservationByKey
        env = ObservationByKey(env, "pixels")
    elif flatten_obs:
        from gym.wrappers import FlattenObservation
        env = FlattenObservation(env)
    from gym.wrappers import TimeLimit
    return TimeLimit(env, max_episode_steps=max_episode_steps)


for difficulty in ['easy', 'medium', 'hard']:
    for domain_name, task_name in ALL_TASKS:
        suffix = difficulty + '-v1' if difficulty else 'v1'
        ID = f'{domain_name.capitalize()}-{task_name}-{suffix}'
        if difficulty:
            distraction_types=('background', 'camera', 'color')
        else:
            distraction_types=None
        register(id=ID,
                 entry_point='distracting_control:make_env',
                 kwargs=dict(domain_name=domain_name,
                             task_name=task_name,
                             difficulty=difficulty,
                             distraction_types=distraction_types,
                             channels_first=True,
                             width=84,
                             height=84,
                             frame_skip=4),
                 )

GDC_IS_REGISTERED = True
