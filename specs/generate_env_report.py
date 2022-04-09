import gym
from cmx import CommonMark


def generate_env_readme():
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
                env = gym.make(f'distracting_control/{domain.capitalize()}-{task}-{difficulty}-v1', from_pixels=True,
                               channels_first=False)
                env.seed(100)
                try:
                    img = env.reset()
                    r.figure(img, src=f"figures/{domain}-{task}-{difficulty}.png", title=difficulty)
                except:
                    doc.print(domain, task, difficulty, f'is not supported.')
                    pass


if __name__ == "__main__":
    generate_env_readme()
