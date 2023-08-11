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

from typing import List

import numpy as np
import torch

from utilities.normalization import DatasetNormalizer


class SequenceDataset(torch.utils.data.Dataset):
    """DataLoader with customized sampler."""

    def __init__(
        self,
        data: dict,
        horizon: int,
        max_traj_length: int,
        normalizer: str = "LimitsNormalizer",
        discrete_action: bool = False,
        use_padding: bool = True,
        use_action: bool = True,
        include_returns: bool = True,
        include_cost_returns: bool = True,
        discount: float = 0.99,
        returns_scale: float = 1.0,
        cost_returns_scale: float = 1.0,
        use_future_masks: bool = False,
    ) -> None:
        self.include_returns = include_returns
        self.include_cost_returns = include_cost_returns
        self.use_action = use_action
        self.use_future_masks = use_future_masks
        self.use_padding = use_padding
        self.max_traj_length = max_traj_length
        self.horizon = horizon
        self.returns_scale = returns_scale
        self.cost_returns_scale = cost_returns_scale
        self.discrete_action = discrete_action
        if discrete_action:
            raise NotImplementedError

        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_traj_length)[:, None]

        self._data = data
        self.normalizer = DatasetNormalizer(
            self._data,
            normalizer,
        )

        self._keys = list(data.keys()).remove("traj_lengths")
        self._indices = self.make_indices()

        self.n_episodes = len(self._data)
        self.normalize()
        print(self._data)

    def __len__(self):
        return len(self._indices)

    def make_indices(self):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """

        indices = []
        for i, traj_length in enumerate(self._data["traj_lengths"]):
            # get `max_start`
            if self.use_future_masks:
                max_start = min(traj_length - 1, self.max_traj_length - 1)
                if not self.use_padding:
                    max_start = min(max_start, traj_length - 1)
            else:
                max_start = min(
                    traj_length, self.max_traj_length - self.horizon + 1
                )
                if not self.use_padding:
                    max_start = min(max_start, traj_length - self.horizon + 1)

            # get `end` and `mask_end` for each `start`
            for start in range(max_start):
                end = start + self.horizon
                if not self.use_padding:
                    mask_end = min(end, traj_length)
                else:
                    mask_end = min(end, self.max_traj_length)
                indices.append((i, start, end, mask_end))
        indices = np.array(indices)
        return indices

    def normalize(self, keys: List[str] = None) -> None:
        """
        normalize fields that will be predicted by the diffusion model
        """
        if keys is None:
            keys = ["observations", "actions"] if self.use_action else ["observations"]

        for key in keys:
            array = self._data[key].reshape(
                self.n_episodes * self.max_traj_length, *self._data[key].shape[2:]
            )
            normed = self.normalizer(array, key)
            self._data[f"normed_{key}"] = normed.reshape(
                self.n_episodes, self.max_traj_length, *self._data[key].shape[2:]
            )

    def get_conditions(self, observations):
        """
        condition on current observation for planning
        """

        return {0: observations[0]}

    def __getitem__(self, idx):
        path_ind, start, end, mask_end = self._indices[idx]

        observations = self._data.normed_observations[path_ind, start:end]
        if self.use_action:
            if self.discrete_action:
                actions = self._data.actions[path_ind, start:end]
            else:
                actions = self._data.normed_actions[path_ind, start:end]

        if mask_end < end:
            observations = np.concatenate(
                [
                    observations,
                    np.zeros(
                        (end - mask_end, observations.shape[-1]),
                        dtype=observations.dtype,
                    ),
                ],
                axis=0,
            )
            if self.use_action:
                actions = np.concatenate(
                    [
                        actions,
                        np.zeros(
                            (end - mask_end, actions.shape[-1]),
                            dtype=actions.dtype,
                        ),
                    ],
                    axis=0,
                )

        masks = np.zeros((observations.shape[0], observations.shape[1]))
        masks[: mask_end - start] = 1.0

        conditions = self.get_conditions(observations)

        ret_dict = dict(samples=observations, conditions=conditions, masks=masks)

        if self.include_returns:
            rewards = self._data.rewards[path_ind, start:]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum(axis=0).squeeze(-1)
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            ret_dict["returns"] = returns

        if self.include_cost_returns:
            costs = self._data.costs[path_ind, start:]
            discounts = self.discounts[: len(costs)]
            cost_returns = (discounts * costs).sum(axis=0).squeeze(-1)
            cost_returns = np.array(
                [cost_returns / self.cost_returns_scale], dtype=np.float32
            )
            ret_dict["cost_returns"] = cost_returns

        if self.use_action:
            ret_dict["actions"] = actions

        return ret_dict


class QLearningDataset(SequenceDataset):
    def make_indices(self):
        assert self.horizon == 1, "QLearningDataset only supports horizon=1"
        return super().make_indices()

    def normalize(self, keys: List[str] = None) -> None:
        """
        normalize fields that will be predicted by the diffusion model
        """

        super().normalize(keys)

        array = self._data["next_observations"].reshape(
            self.n_episodes * self.max_traj_length,
            *self._data["next_observations"].shape[2:],
        )
        normed = self.normalizer(array, "observations")
        self._data["normed_next_observations"] = normed.reshape(
            self.n_episodes,
            self.max_traj_length,
            *self._data["next_observations"].shape[2:],
        )

    def get_conditions(self, observations):
        return {}

    def __getitem__(self, idx):
        path_ind, start, end, mask_end = self._indices[idx]

        observations = self._data.normed_observations[path_ind, start:end].squeeze(0)
        actions = self._data.actions[path_ind, start:end].squeeze(0)
        rewards = self._data.rewards[path_ind, start:end].squeeze(0)
        next_observations = self._data.normed_next_observations[
            path_ind, start:end
        ].squeeze(0)
        dones = self._data.terminals[path_ind, start:end].squeeze(0)

        conditions = self.get_conditions(observations)
        next_conditions = self.get_conditions(next_observations)

        ret_dict = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            conditions=conditions,
            next_conditions=next_conditions,
        )
        return ret_dict
