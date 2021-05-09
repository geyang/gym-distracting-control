import gym


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
