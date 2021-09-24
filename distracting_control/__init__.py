from dm_control.suite import ALL_TASKS
from gym.envs import register


def make_env(flatten_obs=True, from_pixels=False, frame_skip=1, max_episode_steps=1000, **kwargs):
    max_episode_steps /= frame_skip

    from distracting_control.gym_env import DistractingEnv

    env = DistractingEnv(
        from_pixels=from_pixels, frame_skip=frame_skip, **kwargs)
    if from_pixels:
        from .wrappers import ObservationByKey
        env = ObservationByKey(env, "pixels")
    elif flatten_obs:
        from gym.wrappers import FlattenObservation
        env = FlattenObservation(env)
    from gym.wrappers import TimeLimit
    return TimeLimit(env, max_episode_steps=max_episode_steps)


for difficulty in ['easy', 'medium', 'hard', 'intensity']:
    for domain_name, task_name in ALL_TASKS:
        ID = f'{domain_name.capitalize()}-{task_name}-{difficulty}-v1'
        register(id=ID,
                 entry_point='distracting_control:make_env',
                 kwargs=dict(domain_name=domain_name,
                             task_name=task_name,
                             difficulty=None if difficulty == 'intensity' else difficulty,
                             intensity=None,
                             distraction_types=('background', 'camera', 'color'),
                             sample_from_edge=False,
                             channels_first=True,
                             width=84,
                             height=84,
                             frame_skip=4),
                 )

GDC_IS_REGISTERED = True
