import gym
from cmx import CommonMark


def test_max_episode_steps():
    env = gym.make('distracting_control:Walker-walk-easy-v1')
    assert env._max_episode_steps == 250


def test_flat_obs():
    env = gym.make('distracting_control:Walker-walk-easy-v1', frame_skip=4)
    env.env.env.env.observation_spec()
    assert env.reset().shape == (24,)


def test_frame_skip():
    env = gym.make('distracting_control:Walker-walk-easy-v1', from_pixels=True, frame_skip=8)
    assert env._max_episode_steps == 125
    # default channel goes first
    assert env.reset().shape == (3, 84, 84)


def test_channel_first():
    env = gym.make('distracting_control:Walker-walk-easy-v1', from_pixels=True, channels_first=True)
    assert env.reset().shape == (3, 84, 84)


def test_channel_last():
    env = gym.make('distracting_control:Walker-walk-easy-v1', from_pixels=True, frame_skip=8, channels_first=False)
    assert env._max_episode_steps == 125
    assert env.reset().shape == (84, 84, 3)


def test_gray_scale():
    """
    this currently does not support gray scale, because the DMControl rendering function
    only supports RGB, segmentation mask and so on.

    :return:
    """
    pass


def test_all_envs():
    from tqdm import tqdm

    doc = CommonMark('README.md')
    with doc:
        from dm_control.suite import ALL_TASKS
        doc.print(ALL_TASKS)

    with doc:

        for domain, task in tqdm(ALL_TASKS):
            doc @ f"""
            ## `{domain.capitalize()}-{task}`
            """
            r = doc.table().figure_row()
            for difficulty in ['easy', 'medium', 'hard']:
                env = gym.make(f'gdc:{domain.capitalize()}-{task}-{difficulty}-v1', from_pixels=True,
                               channels_first=False)
                env.seed(100)
                try:
                    img = env.reset()
                    r.figure(img, src=f"figures/{domain}-{task}-{difficulty}.png", title=difficulty)
                except:
                    doc.print(domain, task, difficulty, f'is not supported.')
                    pass
