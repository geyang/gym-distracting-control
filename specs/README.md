```python
from dm_control.suite import ALL_TASKS
doc.print(ALL_TASKS)
```

```
(('acrobot', 'swingup'), ('acrobot', 'swingup_sparse'), ('ball_in_cup', 'catch'), ('cartpole', 'balance'), ('cartpole', 'balance_sparse'), ('cartpole', 'swingup'), ('cartpole', 'swingup_sparse'), ('cartpole', 'two_poles'), ('cartpole', 'three_poles'), ('cheetah', 'run'), ('dog', 'stand'), ('dog', 'walk'), ('dog', 'trot'), ('dog', 'run'), ('dog', 'fetch'), ('finger', 'spin'), ('finger', 'turn_easy'), ('finger', 'turn_hard'), ('fish', 'upright'), ('fish', 'swim'), ('hopper', 'stand'), ('hopper', 'hop'), ('humanoid', 'stand'), ('humanoid', 'walk'), ('humanoid', 'run'), ('humanoid', 'run_pure_state'), ('humanoid_CMU', 'stand'), ('humanoid_CMU', 'run'), ('lqr', 'lqr_2_1'), ('lqr', 'lqr_6_2'), ('manipulator', 'bring_ball'), ('manipulator', 'bring_peg'), ('manipulator', 'insert_ball'), ('manipulator', 'insert_peg'), ('pendulum', 'swingup'), ('point_mass', 'easy'), ('point_mass', 'hard'), ('quadruped', 'walk'), ('quadruped', 'run'), ('quadruped', 'escape'), ('quadruped', 'fetch'), ('reacher', 'easy'), ('reacher', 'hard'), ('stacker', 'stack_2'), ('stacker', 'stack_4'), ('swimmer', 'swimmer6'), ('swimmer', 'swimmer15'), ('walker', 'stand'), ('walker', 'walk'), ('walker', 'run'))
```
```python
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
```


## `Acrobot-swingup`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/acrobot-swingup-easy.png](figures/acrobot-swingup-easy.png) | ![figures/acrobot-swingup-medium.png](figures/acrobot-swingup-medium.png) | ![figures/acrobot-swingup-hard.png](figures/acrobot-swingup-hard.png) |


## `Acrobot-swingup_sparse`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/acrobot-swingup_sparse-easy.png](figures/acrobot-swingup_sparse-easy.png) | ![figures/acrobot-swingup_sparse-medium.png](figures/acrobot-swingup_sparse-medium.png) | ![figures/acrobot-swingup_sparse-hard.png](figures/acrobot-swingup_sparse-hard.png) |


## `Ball_in_cup-catch`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/ball_in_cup-catch-easy.png](figures/ball_in_cup-catch-easy.png) | ![figures/ball_in_cup-catch-medium.png](figures/ball_in_cup-catch-medium.png) | ![figures/ball_in_cup-catch-hard.png](figures/ball_in_cup-catch-hard.png) |


## `Cartpole-balance`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/cartpole-balance-easy.png](figures/cartpole-balance-easy.png) | ![figures/cartpole-balance-medium.png](figures/cartpole-balance-medium.png) | ![figures/cartpole-balance-hard.png](figures/cartpole-balance-hard.png) |


## `Cartpole-balance_sparse`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/cartpole-balance_sparse-easy.png](figures/cartpole-balance_sparse-easy.png) | ![figures/cartpole-balance_sparse-medium.png](figures/cartpole-balance_sparse-medium.png) | ![figures/cartpole-balance_sparse-hard.png](figures/cartpole-balance_sparse-hard.png) |


## `Cartpole-swingup`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/cartpole-swingup-easy.png](figures/cartpole-swingup-easy.png) | ![figures/cartpole-swingup-medium.png](figures/cartpole-swingup-medium.png) | ![figures/cartpole-swingup-hard.png](figures/cartpole-swingup-hard.png) |


## `Cartpole-swingup_sparse`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/cartpole-swingup_sparse-easy.png](figures/cartpole-swingup_sparse-easy.png) | ![figures/cartpole-swingup_sparse-medium.png](figures/cartpole-swingup_sparse-medium.png) | ![figures/cartpole-swingup_sparse-hard.png](figures/cartpole-swingup_sparse-hard.png) |


## `Cartpole-two_poles`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/cartpole-two_poles-easy.png](figures/cartpole-two_poles-easy.png) | ![figures/cartpole-two_poles-medium.png](figures/cartpole-two_poles-medium.png) | ![figures/cartpole-two_poles-hard.png](figures/cartpole-two_poles-hard.png) |


## `Cartpole-three_poles`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/cartpole-three_poles-easy.png](figures/cartpole-three_poles-easy.png) | ![figures/cartpole-three_poles-medium.png](figures/cartpole-three_poles-medium.png) | ![figures/cartpole-three_poles-hard.png](figures/cartpole-three_poles-hard.png) |


## `Cheetah-run`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/cheetah-run-easy.png](figures/cheetah-run-easy.png) | ![figures/cheetah-run-medium.png](figures/cheetah-run-medium.png) | ![figures/cheetah-run-hard.png](figures/cheetah-run-hard.png) |


## `Dog-stand`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/dog-stand-easy.png](figures/dog-stand-easy.png) | ![figures/dog-stand-medium.png](figures/dog-stand-medium.png) | ![figures/dog-stand-hard.png](figures/dog-stand-hard.png) |


## `Dog-walk`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/dog-walk-easy.png](figures/dog-walk-easy.png) | ![figures/dog-walk-medium.png](figures/dog-walk-medium.png) | ![figures/dog-walk-hard.png](figures/dog-walk-hard.png) |


## `Dog-trot`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/dog-trot-easy.png](figures/dog-trot-easy.png) | ![figures/dog-trot-medium.png](figures/dog-trot-medium.png) | ![figures/dog-trot-hard.png](figures/dog-trot-hard.png) |


## `Dog-run`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/dog-run-easy.png](figures/dog-run-easy.png) | ![figures/dog-run-medium.png](figures/dog-run-medium.png) | ![figures/dog-run-hard.png](figures/dog-run-hard.png) |


## `Dog-fetch`


```
dog fetch easy is not supported.
dog fetch medium is not supported.
dog fetch hard is not supported.
```


## `Finger-spin`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/finger-spin-easy.png](figures/finger-spin-easy.png) | ![figures/finger-spin-medium.png](figures/finger-spin-medium.png) | ![figures/finger-spin-hard.png](figures/finger-spin-hard.png) |


## `Finger-turn_easy`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/finger-turn_easy-easy.png](figures/finger-turn_easy-easy.png) | ![figures/finger-turn_easy-medium.png](figures/finger-turn_easy-medium.png) | ![figures/finger-turn_easy-hard.png](figures/finger-turn_easy-hard.png) |


## `Finger-turn_hard`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/finger-turn_hard-easy.png](figures/finger-turn_hard-easy.png) | ![figures/finger-turn_hard-medium.png](figures/finger-turn_hard-medium.png) | ![figures/finger-turn_hard-hard.png](figures/finger-turn_hard-hard.png) |


## `Fish-upright`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/fish-upright-easy.png](figures/fish-upright-easy.png) | ![figures/fish-upright-medium.png](figures/fish-upright-medium.png) | ![figures/fish-upright-hard.png](figures/fish-upright-hard.png) |


## `Fish-swim`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/fish-swim-easy.png](figures/fish-swim-easy.png) | ![figures/fish-swim-medium.png](figures/fish-swim-medium.png) | ![figures/fish-swim-hard.png](figures/fish-swim-hard.png) |


## `Hopper-stand`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/hopper-stand-easy.png](figures/hopper-stand-easy.png) | ![figures/hopper-stand-medium.png](figures/hopper-stand-medium.png) | ![figures/hopper-stand-hard.png](figures/hopper-stand-hard.png) |


## `Hopper-hop`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/hopper-hop-easy.png](figures/hopper-hop-easy.png) | ![figures/hopper-hop-medium.png](figures/hopper-hop-medium.png) | ![figures/hopper-hop-hard.png](figures/hopper-hop-hard.png) |


## `Humanoid-stand`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/humanoid-stand-easy.png](figures/humanoid-stand-easy.png) | ![figures/humanoid-stand-medium.png](figures/humanoid-stand-medium.png) | ![figures/humanoid-stand-hard.png](figures/humanoid-stand-hard.png) |


## `Humanoid-walk`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/humanoid-walk-easy.png](figures/humanoid-walk-easy.png) | ![figures/humanoid-walk-medium.png](figures/humanoid-walk-medium.png) | ![figures/humanoid-walk-hard.png](figures/humanoid-walk-hard.png) |


## `Humanoid-run`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/humanoid-run-easy.png](figures/humanoid-run-easy.png) | ![figures/humanoid-run-medium.png](figures/humanoid-run-medium.png) | ![figures/humanoid-run-hard.png](figures/humanoid-run-hard.png) |


## `Humanoid-run_pure_state`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/humanoid-run_pure_state-easy.png](figures/humanoid-run_pure_state-easy.png) | ![figures/humanoid-run_pure_state-medium.png](figures/humanoid-run_pure_state-medium.png) | ![figures/humanoid-run_pure_state-hard.png](figures/humanoid-run_pure_state-hard.png) |


## `Humanoid_cmu-stand`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/humanoid_CMU-stand-easy.png](figures/humanoid_CMU-stand-easy.png) | ![figures/humanoid_CMU-stand-medium.png](figures/humanoid_CMU-stand-medium.png) | ![figures/humanoid_CMU-stand-hard.png](figures/humanoid_CMU-stand-hard.png) |


## `Humanoid_cmu-run`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/humanoid_CMU-run-easy.png](figures/humanoid_CMU-run-easy.png) | ![figures/humanoid_CMU-run-medium.png](figures/humanoid_CMU-run-medium.png) | ![figures/humanoid_CMU-run-hard.png](figures/humanoid_CMU-run-hard.png) |


## `Lqr-lqr_2_1`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/lqr-lqr_2_1-easy.png](figures/lqr-lqr_2_1-easy.png) | ![figures/lqr-lqr_2_1-medium.png](figures/lqr-lqr_2_1-medium.png) | ![figures/lqr-lqr_2_1-hard.png](figures/lqr-lqr_2_1-hard.png) |


## `Lqr-lqr_6_2`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/lqr-lqr_6_2-easy.png](figures/lqr-lqr_6_2-easy.png) | ![figures/lqr-lqr_6_2-medium.png](figures/lqr-lqr_6_2-medium.png) | ![figures/lqr-lqr_6_2-hard.png](figures/lqr-lqr_6_2-hard.png) |


## `Manipulator-bring_ball`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/manipulator-bring_ball-easy.png](figures/manipulator-bring_ball-easy.png) | ![figures/manipulator-bring_ball-medium.png](figures/manipulator-bring_ball-medium.png) | ![figures/manipulator-bring_ball-hard.png](figures/manipulator-bring_ball-hard.png) |


## `Manipulator-bring_peg`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/manipulator-bring_peg-easy.png](figures/manipulator-bring_peg-easy.png) | ![figures/manipulator-bring_peg-medium.png](figures/manipulator-bring_peg-medium.png) | ![figures/manipulator-bring_peg-hard.png](figures/manipulator-bring_peg-hard.png) |


## `Manipulator-insert_ball`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/manipulator-insert_ball-easy.png](figures/manipulator-insert_ball-easy.png) | ![figures/manipulator-insert_ball-medium.png](figures/manipulator-insert_ball-medium.png) | ![figures/manipulator-insert_ball-hard.png](figures/manipulator-insert_ball-hard.png) |


## `Manipulator-insert_peg`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/manipulator-insert_peg-easy.png](figures/manipulator-insert_peg-easy.png) | ![figures/manipulator-insert_peg-medium.png](figures/manipulator-insert_peg-medium.png) | ![figures/manipulator-insert_peg-hard.png](figures/manipulator-insert_peg-hard.png) |


## `Pendulum-swingup`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/pendulum-swingup-easy.png](figures/pendulum-swingup-easy.png) | ![figures/pendulum-swingup-medium.png](figures/pendulum-swingup-medium.png) | ![figures/pendulum-swingup-hard.png](figures/pendulum-swingup-hard.png) |


## `Point_mass-easy`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/point_mass-easy-easy.png](figures/point_mass-easy-easy.png) | ![figures/point_mass-easy-medium.png](figures/point_mass-easy-medium.png) | ![figures/point_mass-easy-hard.png](figures/point_mass-easy-hard.png) |


## `Point_mass-hard`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/point_mass-hard-easy.png](figures/point_mass-hard-easy.png) | ![figures/point_mass-hard-medium.png](figures/point_mass-hard-medium.png) | ![figures/point_mass-hard-hard.png](figures/point_mass-hard-hard.png) |


## `Quadruped-walk`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/quadruped-walk-easy.png](figures/quadruped-walk-easy.png) | ![figures/quadruped-walk-medium.png](figures/quadruped-walk-medium.png) | ![figures/quadruped-walk-hard.png](figures/quadruped-walk-hard.png) |


## `Quadruped-run`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/quadruped-run-easy.png](figures/quadruped-run-easy.png) | ![figures/quadruped-run-medium.png](figures/quadruped-run-medium.png) | ![figures/quadruped-run-hard.png](figures/quadruped-run-hard.png) |


## `Quadruped-escape`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/quadruped-escape-easy.png](figures/quadruped-escape-easy.png) | ![figures/quadruped-escape-medium.png](figures/quadruped-escape-medium.png) | ![figures/quadruped-escape-hard.png](figures/quadruped-escape-hard.png) |


## `Quadruped-fetch`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/quadruped-fetch-easy.png](figures/quadruped-fetch-easy.png) | ![figures/quadruped-fetch-medium.png](figures/quadruped-fetch-medium.png) | ![figures/quadruped-fetch-hard.png](figures/quadruped-fetch-hard.png) |


## `Reacher-easy`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/reacher-easy-easy.png](figures/reacher-easy-easy.png) | ![figures/reacher-easy-medium.png](figures/reacher-easy-medium.png) | ![figures/reacher-easy-hard.png](figures/reacher-easy-hard.png) |


## `Reacher-hard`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/reacher-hard-easy.png](figures/reacher-hard-easy.png) | ![figures/reacher-hard-medium.png](figures/reacher-hard-medium.png) | ![figures/reacher-hard-hard.png](figures/reacher-hard-hard.png) |


## `Stacker-stack_2`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/stacker-stack_2-easy.png](figures/stacker-stack_2-easy.png) | ![figures/stacker-stack_2-medium.png](figures/stacker-stack_2-medium.png) | ![figures/stacker-stack_2-hard.png](figures/stacker-stack_2-hard.png) |


## `Stacker-stack_4`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/stacker-stack_4-easy.png](figures/stacker-stack_4-easy.png) | ![figures/stacker-stack_4-medium.png](figures/stacker-stack_4-medium.png) | ![figures/stacker-stack_4-hard.png](figures/stacker-stack_4-hard.png) |


## `Swimmer-swimmer6`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/swimmer-swimmer6-easy.png](figures/swimmer-swimmer6-easy.png) | ![figures/swimmer-swimmer6-medium.png](figures/swimmer-swimmer6-medium.png) | ![figures/swimmer-swimmer6-hard.png](figures/swimmer-swimmer6-hard.png) |


## `Swimmer-swimmer15`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/swimmer-swimmer15-easy.png](figures/swimmer-swimmer15-easy.png) | ![figures/swimmer-swimmer15-medium.png](figures/swimmer-swimmer15-medium.png) | ![figures/swimmer-swimmer15-hard.png](figures/swimmer-swimmer15-hard.png) |


## `Walker-stand`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/walker-stand-easy.png](figures/walker-stand-easy.png) | ![figures/walker-stand-medium.png](figures/walker-stand-medium.png) | ![figures/walker-stand-hard.png](figures/walker-stand-hard.png) |


## `Walker-walk`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/walker-walk-easy.png](figures/walker-walk-easy.png) | ![figures/walker-walk-medium.png](figures/walker-walk-medium.png) | ![figures/walker-walk-hard.png](figures/walker-walk-hard.png) |


## `Walker-run`

| **easy** | **medium** | **hard** |
|:--------:|:----------:|:--------:|
| ![figures/walker-run-easy.png](figures/walker-run-easy.png) | ![figures/walker-run-medium.png](figures/walker-run-medium.png) | ![figures/walker-run-hard.png](figures/walker-run-hard.png) |
