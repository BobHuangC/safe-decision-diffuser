import numpy as np

from utilities.data_utils import atleast_nd


def clip_actions(dataset, clip_to_eps: bool = True, eps: float = 1e-5):
    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)
    return dataset


def split_to_trajs(
    dataset,
    use_cost: bool = False,
    norm_reward: bool = False,
    norm_cost: bool = False,
):
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

    trajs = [[]]
    for i in range(len(dataset["observations"])):
        if use_cost:
            trajs[-1].append(
                (
                    dataset["observations"][i],
                    dataset["actions"][i],
                    dataset["rewards"][i],
                    dones_float[i],
                    dataset["next_observations"][i],
                    dataset["costs"][i],
                )
            )
        else:
            trajs[-1].append(
                (
                    dataset["observations"][i],
                    dataset["actions"][i],
                    dataset["rewards"][i],
                    dones_float[i],
                    dataset["next_observations"][i],
                )
            )
        if dones_float[i] == 1.0 and i + 1 < len(dataset["observations"]):
            trajs.append([])

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, *_ in traj:
            episode_return += rew
        return episode_return

    def compute_cost_returns(traj):
        episode_cost_return = 0
        for *_, cost in traj:
            episode_cost_return += cost
        return episode_cost_return

    if norm_reward:
        returns = [compute_returns(traj) for traj in trajs]
        norm = (max(returns) - min(returns)) / 1000
        for traj in trajs:
            for i, ts in enumerate(traj):
                traj[i] = ts[:2] + (ts[2] / norm,) + ts[3:]

    if norm_cost:
        assert use_cost is True, "Cannot normalize cost without using cost"
        cost_returns = [compute_cost_returns(traj) for traj in trajs]
        norm = (max(cost_returns) - min(cost_returns)) / 1000
        for traj in trajs:
            for i, ts in enumerate(traj):
                traj[i] = ts[:-1] + (ts[-1] / norm,)

    return trajs


def pad_trajs_to_dataset(
    trajs,
    max_traj_length: int,
    use_cost: bool = False,
    termination_penalty: float = None,
    include_next_obs: bool = False,
):
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
    if use_cost:
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
        dataset["terminals"][idx, :traj_length] = np.stack([bool(ts[3]) for ts in traj], axis=0)
        dataset["dones_float"][idx, :traj_length] = np.stack([ts[3] for ts in traj], axis=0)
        if use_cost:
            dataset["costs"][idx, :traj_length] = atleast_nd(
                np.stack([ts[5] for ts in traj], axis=0), n=2,
            )
        if include_next_obs:
            dataset["next_observations"][idx, :traj_length] = atleast_nd(
                np.stack([ts[4] for ts in traj], axis=0), n=2,
            )
        if dataset["terminals"][idx].any() and termination_penalty is not None:
            dataset["rewards"][idx, traj_length - 1] += termination_penalty

    return dataset
