# The Distracting Control Suite

This is a packaged version of the `distracting_control` suite from *Stone et al*. We provide OpenAI gym bindings, to make the original code base easier to use.

## Getting Started

```bash
pip install distracting_control
```

Then in your python script:

```python
import gym

env = gyn.make('gdc:Hopper-hop-easy-v1', from_pixel=True)
obs = env.reset()
doc.figure(obs, "figures/hopper_readme.png")
```

## Detailed API

Take a look at the test file in the [./specs](./specs) folder, and the source code. DeepMind control has a lot of low level binding burried in the source code.

```python
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
```



### Important Changes from *Stone et al*

1. [planned] remove tensorflow dependency
2. [planned] increase ground floor transparency in `Hopper`

## Original README

`distracting_control` extends `dm_control` with static or dynamic visual
distractions in the form of changing colors, backgrounds, and camera poses. 
Details and experimental results can be found in our
[paper](https://arxiv.org/pdf/2101.02722.pdf).

## Requirements and Installation

* Clone this repository
* `sh run.sh`
* Follow the instructions and install
[dm_control](https://github.com/deepmind/dm_control#requirements-and-installation). Make sure you setup your MuJoCo keys correctly.
* Download the [DAVIS 2017
  dataset](https://davischallenge.org/davis2017/code.html). Make sure to select the 2017 TrainVal - Images and Annotations (480p). The training images will be used as distracting backgrounds.

## Instructions

* You can run the `distracting_control_demo` to generate sample images of the
  different tasks at different difficulties:

  ```
  python distracting_control_demo --davis_path=$HOME/DAVIS/JPEGImages/480p/
  --output_dir=/tmp/distrtacting_control_demo
  ```
* As seen from the demo to generate an instance of the environment you simply
  need to import the suite and use `suite.load` while specifying the
  `dm_control` domain and task, then choosing a difficulty and providing the
  dataset_path.

* Note the environment follows the dm_control environment APIs.

## Paper

If you use this code, please cite the accompanying [paper](https://arxiv.org/pdf/2101.02722.pdf) as:

```
@article{stone2021distracting,
      title={The Distracting Control Suite -- A Challenging Benchmark for Reinforcement Learning from Pixels}, 
      author={Austin Stone and Oscar Ramirez and Kurt Konolige and Rico Jonschkowski},
      year={2021},
      journal={arXiv preprint arXiv:2101.02722},
}
```

## Disclaimer

This is not an official Google product.

