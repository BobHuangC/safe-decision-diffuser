# Copyright 2023 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils to generate trajectory based dataset with multi-step reward."""

from typing import List
import dsrl
import gymnasium
import numpy as np
from tqdm import tqdm

from utilities.deprecation import deprecated
from utilities.data_utils import atleast_nd


def split_into_trajectories(
    observations, actions, rewards, masks, dones_float, next_observations, costs
):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                masks[i],
                dones_float[i],
                next_observations[i],
                costs[i],
            )
        )
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def get_dsrl_dataset(
    env,
    max_traj_length: int,
    norm_reward: bool = False,
    norm_cost: bool = False,
    termination_penalty: float = None,
    include_next_obs: bool = False,
    cost_termination_penalty: float = None,
):
    # set `sorting=True` to sort the dataset according to trajectory return
    trajs = get_dsrl_trajs(env, norm_reward=norm_reward)
    n_trajs = len(trajs)

    dataset = {}
    obs_dim, act_dim = trajs[0][0][0].shape[0], trajs[0][0][1].shape[0]
    dataset["observations"] = np.zeros(
        (n_trajs, max_traj_length, obs_dim), dtype=np.float32
    )
    dataset["actions"] = np.zeros(
        (n_trajs, max_traj_length, act_dim), dtype=np.float32
    )
    dataset["rewards"] = np.zeros((n_trajs, max_traj_length, 1), dtype=np.float32)
    dataset["terminals"] = np.zeros((n_trajs, max_traj_length), dtype=np.float32)
    dataset["dones_float"] = np.zeros((n_trajs, max_traj_length), dtype=np.float32)
    dataset["traj_lengths"] = np.zeros((n_trajs,), dtype=np.int32)
    dataset["costs"] = np.zeros((n_trajs, max_traj_length, 1), dtype=np.float32)
    if include_next_obs:
        dataset["next_observations"] = np.zeros((n_trajs, max_traj_length, obs_dim), dtype=np.float32)

    for idx, traj in enumerate(trajs):
        traj_length = len(traj)
        dataset["traj_lengths"][idx] = traj_length
        dataset["observations"][idx, :traj_length] = atleast_nd(
            np.stack([ts[0] for ts in traj], axis=0), n=2,
        )
        dataset["actions"][idx, :traj_length] = atleast_nd(
            np.stack([ts[1] for ts in traj], axis=0), n=2,
        )
        dataset["rewards"][idx, :traj_length] = atleast_nd(
            np.stack([ts[2] for ts in traj], axis=0), n=2,
        )
        dataset["terminals"][idx, :traj_length] = np.stack([bool(1 - ts[3]) for ts in traj], axis=0)
        dataset["dones_float"][idx, :traj_length] = np.stack([ts[4] for ts in traj], axis=0)
        if include_next_obs:
            dataset["next_observations"][idx, :traj_length] = atleast_nd(
                np.stack([ts[5] for ts in traj], axis=0), n=2,
            )
        if dataset["terminals"][idx].any() and termination_penalty is not None:
            dataset["rewards"][idx, traj_length - 1] += termination_penalty
        if dataset["terminals"][idx].any() and cost_termination_penalty is not None:
            dataset["costs"][idx, traj_length - 1] += cost_termination_penalty

    return Dataset(**dataset)


class Dataset:
    __initialized = False

    def __init__(
        self,
        required_keys: List[str] = ["observations", "actions", "rewards", "dones_float"],
        verbose: bool = True,
        **kwargs,
    ):
        self._dict = {}
        for k, v in kwargs.items():
            self[k] = v

        self.required_keys = []
        self.extra_keys = []
        for k in self.keys():
            if k in required_keys:
                self.required_keys.append(k)
            else:
                self.extra_keys.append(k)
        assert set(self.required_keys) == set(required_keys), f"Missing keys: {set(required_keys) - set(self.required_keys)}"
        if verbose:
            print("[ data/dataset.py ] Dataset: get required keys:", self.required_keys)
            print("[ data/dataset.py ] Dataset: get extra keys:", self.extra_keys)
        self.__initialized = True

    def __setattr__(self, k, v):
        if self.__initialized and k not in self._dict.keys():
            raise AttributeError(f"Cannot add new attributes to Dataset: {k}")
        else:
            object.__setattr__(self, k, v)

    def keys(self):
        return self._dict.keys()

    def items(self):
        return {k: v for k, v in self._dict.items() if k != "traj_lengths"}.items()

    def __len__(self):
        return self._dict["observations"].shape[0]

    def __repr__(self):
        return "[ data/dataset.py ] Dataset:\n" + "\n".join(
            f"    {key}: {val.shape}" for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        setattr(self, key, val)


class DSRLDataset(Dataset):
    def __init__(self, env: gymnasium.Env, clip_to_eps: bool = True, eps: float = 1e-5, **kwargs):
        self.raw_dataset = dataset = env.get_dataset()

        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(
            observations=dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminals"].astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            costs=dataset["costs"].astype(np.float32),
            **kwargs,
        )


def compute_returns(traj):
    episode_return = 0
    for _, _, rew, _, _, _, _ in traj:
        episode_return += rew
    return episode_return


def compute_cost_returns(traj):
    episode_cost_return = 0
    for _, _, _, _, _, _, _, cost in traj:
        episode_cost_return += cost
    return episode_cost_return


def get_dsrl_trajs(env, sorting: bool = False, norm_reward: bool = False):
    env = gym.make(env) if isinstance(env, str) else env
    dataset = DSRLDataset(env, verbose=False)
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
        dataset.costs,
    )
    if sorting:
        trajs.sort(key=compute_returns)

    if norm_reward:
        returns = [compute_returns(traj) for traj in trajs]
        norm = (max(returns) - min(returns)) / 1000
        for traj in tqdm(trajs):
            for i, ts in enumerate(traj):
                traj[i] = ts[:2] + (ts[2] / norm,) + ts[3:]

    # NOTE: this raw_dataset is not sorted
    # return trajs, dataset.raw_dataset
    return trajs


def nstep_reward_prefix(rewards, nstep=5, gamma=0.9):
    gammas = np.array([gamma**i for i in range(nstep)])
    nstep_rewards = np.convolve(rewards, gammas)[nstep - 1 :]
    return nstep_rewards


def nstep_cost_prefix(costs, nstep=5, gamma=0.9):
    gammas = np.array([gamma**i for i in range(nstep)])
    nstep_costs = np.convolve(costs, gammas)[nstep - 1 :]
    return nstep_costs


@deprecated(replacement="get_traj_dataset")
def get_nstep_dsrl_dataset(env, nstep=5, gamma=0.9, sorting=True, norm_reward=False):
    gammas = np.array([gamma**i for i in range(nstep)])
    trajs = get_dsrl_trajs(env, sorting, norm_reward)

    obss, acts, terms, next_obss, nstep_rews, dones_float, nstep_costs = [], [], [], [], [], [], []
    for traj in trajs:
        L = len(traj)
        rewards = np.array([ts[2] for ts in traj])
        cum_rewards = np.convolve(rewards, gammas)[nstep - 1 :]
        nstep_rews.append(cum_rewards)
        next_obss.extend([traj[min(i + nstep - 1, L - 1)][-1] for i in range(L)])
        obss.extend([traj[i][0] for i in range(L)])
        acts.extend([traj[i][1] for i in range(L)])
        terms.extend([bool(1 - traj[i][3]) for i in range(L)])
        dones_float.extend(traj[i][4] for i in range(L))
        costs = np.array([ts[6] for ts in traj])
        cum_costs = np.convolve(costs, gammas)[nstep - 1 :]
        nstep_costs.append(cum_costs)

    dataset = {}
    dataset["observations"] = np.stack(obss)
    dataset["actions"] = np.stack(acts)
    dataset["next_observations"] = np.stack(next_obss)
    dataset["rewards"] = np.concatenate(nstep_rews)
    dataset["terminals"] = np.stack(terms)
    dataset["dones_float"] = np.stack(dones_float)
    dataset["costs"] = np.concatenate(nstep_costs)

    return dataset
