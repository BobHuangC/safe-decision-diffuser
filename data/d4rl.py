from functools import partial

import d4rl
import gym

from utilities.utils import compose

from .dataset import Dataset
from .preprocess import clip_actions, pad_trajs_to_dataset, split_to_trajs


def get_dataset(
    env,
    max_traj_length: int,
    norm_reward: bool = False,
    termination_penalty: float = None,
    include_next_obs: bool = False,
):
    preprocess_fn = compose(
        partial(
            pad_trajs_to_dataset,
            max_traj_length=max_traj_length,
            use_cost=False,
            termination_penalty=termination_penalty,
            include_next_obs=include_next_obs,
        ),
        partial(
            split_to_trajs,
            norm_reward=norm_reward,
            use_cost=False,
        ),
        clip_actions,
    )
    return D4RLDataset(env, preprocess_fn=preprocess_fn)


class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, preprocess_fn, **kwargs):
        self.raw_dataset = dataset = d4rl.qlearning_dataset(env)
        data_dict = preprocess_fn(dataset)
        super().__init__(**data_dict, **kwargs)
