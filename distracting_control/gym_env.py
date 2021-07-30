import gym
import numpy as np
from dm_env import specs
from gym import spaces

from distracting_control import suite


def convert_dm_control_to_gym_space(dm_control_space, **kwargs):
    r"""Convert dm_control space to gym space. """
    if isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=dm_control_space.minimum,
                           high=dm_control_space.maximum,
                           dtype=dm_control_space.dtype)
        assert space.shape == dm_control_space.shape
        return space
    elif isinstance(dm_control_space, specs.Array) and not isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=-float('inf'),
                           high=float('inf'),
                           shape=dm_control_space.shape,
                           dtype=dm_control_space.dtype)
        return space
    elif isinstance(dm_control_space, dict):
        kwargs.update(
            {key: convert_dm_control_to_gym_space(value)
             for key, value in dm_control_space.items()}
        )
        space = spaces.Dict(kwargs)
        return space


def extract_min_max(s):
    assert s.dtype == np.float64 or s.dtype == np.float32
    dim = np.int(np.prod(s.shape))
    if type(s) == specs.Array:
        bound = np.inf * np.ones(dim, dtype=np.float32)
        return -bound, bound
    elif type(s) == specs.BoundedArray:
        zeros = np.zeros(dim, dtype=np.float32)
        return s.minimum + zeros, s.maximum + zeros


class DistractingEnv(gym.Env):
    def __init__(self,
                 domain_name,
                 task_name,
                 difficulty,
                 dynamic=False,
                 background_data_path=None,
                 background_kwargs=None,
                 background_dataset_videos="train",

                 from_pixels=False,

                 camera_kwargs=None,
                 color_kwargs=None,
                 task_kwargs=None,
                 environment_kwargs=None,
                 visualize_reward=False,
                 pixels_observation_key="pixels",

                 height=84,
                 width=84,
                 camera_id=0,

                 frame_skip=1,
                 channels_first=True,
                 warmstart=True,  # info: https://github.com/deepmind/dm_control/issues/64
                 no_gravity=False,
                 non_newtonian=False,
                 skip_start=None,  # useful in Manipulator for letting things settle
                 ):
        """

        :param domain_name:
        :param task_name:
        :param difficulty:
        :param dynamic:
        :param background_data_path:
        :param background_kwargs:
        :param background_dataset_videos:
        :param from_pixels:
        :param camera_kwargs:
        :param color_kwargs:
        :param task_kwargs:
        :param environment_kwargs:
        :param visualize_reward:
        :param pixels_observation_key: usually do not change it from `pixels`. None is not supported.
        :param height:
        :param width:
        :param camera_id:
        :param frame_skip:
        :param channels_first:
        :param warmstart:
        :param no_gravity:
        :param non_newtonian:
        :param skip_start:
        """

        self.render_kwargs = dict(
            height=height,
            width=width,
            camera_id=camera_id,
        )

        self.env = suite.load(domain_name,
                              task_name,
                              difficulty,
                              dynamic=dynamic,

                              # distractor kwargs
                              background_dataset_path=background_data_path,
                              background_dataset_videos=background_dataset_videos,
                              background_kwargs=background_kwargs,
                              camera_kwargs=camera_kwargs,
                              color_kwargs=color_kwargs,

                              # original
                              task_kwargs=task_kwargs,
                              environment_kwargs=environment_kwargs,
                              visualize_reward=visualize_reward,

                              render_kwargs=self.render_kwargs,
                              from_pixels=from_pixels,
                              pixels_observation_key=pixels_observation_key,
                              )
        self.pixels_observation_key = pixels_observation_key
        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': round(1.0 / self.env.control_timestep())}

        self.observation_space = convert_dm_control_to_gym_space(self.env.observation_spec())
        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())
        self.viewer = None

        self.from_pixels = from_pixels
        self.frame_skip = frame_skip
        self.channels_first = channels_first
        if not warmstart:
            self.env.physics.data.qacc_warmstart[:] = 0
        self.no_gravity = no_gravity
        self.non_newtonian = non_newtonian

        if self.no_gravity:  # info: this removes gravity.
            self.turn_off_gravity()

        self.skip_start = skip_start

    def turn_off_gravity(self):
        # note: specifically for manipulator, lets the object fall.
        self.env.physisc.body_mass[:-2] = 0

    def seed(self, seed=None):
        self.action_space.seed(seed)
        return self.env.task.random.seed(seed)

    def set_state(self, state):
        # note: missing the goal positions.
        # self.env.physics.
        self.env.physics.set_state(state)
        self.step([0])

    def step(self, action):
        reward = 0
        discount = 1
        for i in range(self.frame_skip):
            ts = self.env.step(action)
            if self.non_newtonian:  # zero velocity if non newtonian
                self.env.physics.data.qvel[:] = 0
            reward += ts.reward or 0
            discount *= ts.discount
            done = ts.last()
            if done:
                break

        sim_state = self.env.physics.get_state().copy()

        obs = ts.observation

        if self.pixels_observation_key in obs and self.channels_first:
            obs[self.pixels_observation_key] = obs[self.pixels_observation_key].transpose([2, 0, 1])

        return obs, reward, done, dict(sim_state=sim_state)

    def reset(self):
        obs = self.env.reset().observation
        for i in range(self.skip_start or 0):
            obs = self.env.step([0]).observation

        if self.pixels_observation_key in obs and self.channels_first:
            obs[self.pixels_observation_key] = obs[self.pixels_observation_key].transpose([2, 0, 1])

        return obs

    def render(self, mode='human', height=None, width=None, camera_id=0, **kwargs):
        img = self.env.physics.render(
            width=self.render_kwargs['width'] if width is None else width,
            height=self.render_kwargs['height'] if height is None else height,
            camera_id=self.render_kwargs['camera_id'] if camera_id is None else camera_id,
            **kwargs)
        if mode in ['rgb', 'rgb_array']:
            return img.astype(np.uint8)
        elif mode in ['gray', 'grey']:
            return img.mean(axis=-1, keepdims=True).astype(np.uint8)
        elif mode == 'notebook':
            from IPython.display import display
            from PIL import Image
            img = Image.fromarray(img, "RGB")
            display(img)
            return img
        elif mode == 'human':
            from PIL import Image
            return Image.fromarray(img)
        else:
            raise NotImplementedError(f"`{mode}` mode is not implemented")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()
