from functools import partial

import gymnasium

from utilities.utils import compose

from .dataset import Dataset
from .preprocess import (
    clip_actions,
    pad_trajs_to_dataset,
    split_to_trajs,
    data_augmentation,
)


def get_dataset(
    env,
    max_traj_length: int,
    use_cost: bool = True,
    norm_reward: bool = False,
    norm_cost: bool = False,
    termination_penalty: float = None,
    include_next_obs: bool = False,
    augmentation_method: str = "",
    augment_percent: float = 0.3,
    deg: int = 3, 
    max_rew_decrease: float = 1, 
    beta: float = 1, 
    max_reward: float = 1000.0, 
    min_reward: float = 0.0, 
    aug_rmin: float = 0, 
    aug_rmax: float = 600, 
    aug_cmin: float = 5, 
    aug_cmax: float = 50, 
    cgap: float = 5, 
    rstd: float = 1, 
    cstd: float = 0.25,
    rmin:float = 0, 
    cost_bins:float = 60, 
    max_num_per_bin:int =1
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
            data_augmentation,
            augmentation_method=augmentation_method,
            augment_percent = augment_percent,
            deg = deg, 
            max_rew_decrease = max_rew_decrease, 
            beta = beta, 
            max_reward = max_reward, 
            min_reward = min_reward, 
            aug_rmin = aug_rmin, 
            aug_rmax = aug_rmax, 
            aug_cmin = aug_cmin, 
            aug_cmax = aug_cmax, 
            cgap = cgap, 
            rstd = rstd, 
            cstd = cstd,
            rmin = rmin, 
            cost_bins = cost_bins, 
            max_num_per_bin = max_num_per_bin
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
