from functools import partial

import gymnasium

from utilities.utils import compose

from .dataset import Dataset
from .preprocess import (
    clip_actions,
    pareto_augmentation,
    select_optimal_trajectory,
    pad_trajs_to_dataset,
    split_to_trajs,
    compute_discounted_returns,
)


def get_dataset(
    env,
    max_traj_length: int,
    discount: float = 1.0,
    cost_discount: float = 1.0,
    termination_penalty: float = None,
    include_next_obs: bool = False,
    pareto_optimal_only: bool = False,
    aug_percent: float = 0.3,
    deg: int = 3,
    max_rew_decrease: float = 1.0,
    beta: float = 1.0,
    max_reward: float = 1000.0,
    min_reward: float = 0.0,
    rmin: float = 0.0,
    cost_bins: int = 60,
    max_num_per_bin: int = 1,
    clip_to_eps: bool = False,
):
    preprocess_fn = compose(
        partial(
            pad_trajs_to_dataset,
            max_traj_length=max_traj_length,
            include_next_obs=include_next_obs,
        ),
        partial(
            select_optimal_trajectory,
            rmin=rmin,
            cost_bins=cost_bins,
            max_num_per_bin=max_num_per_bin,
        )
        if pareto_optimal_only
        else partial(
            pareto_augmentation,
            aug_percent=aug_percent,
            deg=deg,
            max_rew_decrease=max_rew_decrease,
            beta=beta,
            max_reward=max_reward,
            min_reward=min_reward,
        ),
        partial(
            compute_discounted_returns,
            discount=discount,
            cost_discount=cost_discount,
            termination_penalty=termination_penalty,
        ),
        partial(
            split_to_trajs,
            use_timeouts=True,
        ),
        partial(
            clip_actions,
            clip_to_eps=clip_to_eps,
        ),
    )
    return DSRLDataset(env, preprocess_fn=preprocess_fn)


class DSRLDataset(Dataset):
    def __init__(self, env: gymnasium.Env, preprocess_fn, **kwargs):
        self.raw_dataset = dataset = env.get_dataset()
        data_dict = preprocess_fn(dataset)
        super().__init__(**data_dict, **kwargs)
