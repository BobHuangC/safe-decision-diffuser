from functools import partial

import gymnasium

from utilities.utils import compose

from .dataset import Dataset
from .preprocess import clip_actions, pad_trajs_to_dataset, split_to_trajs


def get_dataset(
    env,
    max_traj_length: int,
    use_cost: bool = True,
    norm_reward: bool = False,
    norm_cost: bool = False,
    termination_penalty: float = None,
    include_next_obs: bool = False,
):
    preprocess_fn = compose(
        partial(
            pad_trajs_to_dataset,
            max_traj_length=max_traj_length,
            use_cost=use_cost,
            termination_penalty=termination_penalty,
            include_next_obs=include_next_obs,
        ),
        partial(
            split_to_trajs,
            use_cost=use_cost,
            norm_reward=norm_reward,
            norm_cost=norm_cost,
        ),
        clip_actions,
    )
    return DSRLDataset(env, preprocess_fn=preprocess_fn)


class DSRLDataset(Dataset):
    def __init__(self, env: gymnasium.Env, preprocess_fn, **kwargs):
        self.raw_dataset = dataset = env.get_dataset()
        data_dict = preprocess_fn(dataset)
        super().__init__(**data_dict, **kwargs)
