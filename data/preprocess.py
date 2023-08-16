import numpy as np
import copy
import heapq
import random
from collections import Counter, defaultdict
import oapackage

from utilities.data_utils import atleast_nd


def clip_actions(dataset, clip_to_eps: bool = True, eps: float = 1e-5):
    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)
    return dataset


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


def split_to_trajs(
    dataset,
    use_cost: bool = False,
    norm_reward: bool = False,
    norm_cost: bool = False,
):
    dones_float = np.zeros_like(dataset["rewards"])  # truncated and terminal
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
                    dataset["terminals"][i],
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
                    dataset["terminals"][i],
                    dataset["next_observations"][i],
                )
            )
        if dones_float[i] == 1.0 and i + 1 < len(dataset["observations"]):
            trajs.append([])

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
    dataset["actions"] = np.zeros((n_trajs, max_traj_length, act_dim), dtype=np.float32)
    dataset["rewards"] = np.zeros((n_trajs, max_traj_length), dtype=np.float32)
    dataset["terminals"] = np.zeros((n_trajs, max_traj_length), dtype=np.float32)
    dataset["dones_float"] = np.zeros((n_trajs, max_traj_length), dtype=np.float32)
    dataset["traj_lengths"] = np.zeros((n_trajs,), dtype=np.int32)
    if use_cost:
        dataset["costs"] = np.zeros((n_trajs, max_traj_length), dtype=np.float32)
    if include_next_obs:
        dataset["next_observations"] = np.zeros(
            (n_trajs, max_traj_length, obs_dim), dtype=np.float32
        )

    for idx, traj in enumerate(trajs):
        traj_length = len(traj)
        dataset["traj_lengths"][idx] = traj_length
        dataset["observations"][idx, :traj_length] = atleast_nd(
            np.stack([ts[0] for ts in traj], axis=0),
            n=2,
        )
        dataset["actions"][idx, :traj_length] = atleast_nd(
            np.stack([ts[1] for ts in traj], axis=0),
            n=2,
        )
        dataset["rewards"][idx, :traj_length] = np.stack([ts[2] for ts in traj], axis=0)
        dataset["dones_float"][idx, :traj_length] = np.stack(
            [ts[3] for ts in traj], axis=0
        )
        dataset["terminals"][idx, :traj_length] = np.stack(
            [bool(ts[4]) for ts in traj], axis=0
        )
        if include_next_obs:
            dataset["next_observations"][idx, :traj_length] = atleast_nd(
                np.stack([ts[5] for ts in traj], axis=0),
                n=2,
            )
        if use_cost:
            dataset["costs"][idx, :traj_length] = np.stack(
                [ts[6] for ts in traj], axis=0
            )
        if dataset["terminals"][idx].any() and termination_penalty is not None:
            dataset["rewards"][idx, traj_length - 1] += termination_penalty

    return dataset


def compute_discounted_cumsum_returns(traj, gamma: float) -> np.ndarray:
    """
    Calculate the discounted cumulative reward sum of traj
    """
    cumsum = np.zeros(len(traj))
    cumsum[-1] = traj[-1][2]
    for t in reversed(range(cumsum.shape[0] - 1)):
        cumsum[t] = traj[t][2] + gamma * cumsum[t + 1]
    return cumsum


def compute_discounted_cumsum_cost_returns(traj, gamma: float) -> np.ndarray:
    """
    Calculate the discounted cumulative cost sum of traj
    """
    cumsum = np.zeros(len(traj))
    cumsum[-1] = traj[-1][-1]
    for t in reversed(range(cumsum.shape[0] - 1)):
        cumsum[t] = traj[t][-1] + gamma * cumsum[t + 1]
    return cumsum


def grid_filter(
    x,
    y,
    xmin=-np.inf,
    xmax=np.inf,
    ymin=-np.inf,
    ymax=np.inf,
    xbins=10,
    ybins=10,
    max_num_per_bin=10,
    min_num_per_bin=1,
):
    xmin, xmax = max(min(x), xmin), min(max(x), xmax)
    ymin, ymax = max(min(y), ymin), min(max(y), ymax)
    xbin_step = (xmax - xmin) / xbins
    ybin_step = (ymax - ymin) / ybins
    # the key is x y bin index, the value is a list of indices
    bin_hashmap = defaultdict(list)
    for i in range(len(x)):
        if x[i] < xmin or x[i] > xmax or y[i] < ymin or y[i] > ymax:
            continue
        x_bin_idx = (x[i] - xmin) // xbin_step
        y_bin_idx = (y[i] - ymin) // ybin_step
        bin_hashmap[(x_bin_idx, y_bin_idx)].append(i)
    # start filtering
    indices = []
    for v in bin_hashmap.values():
        if len(v) > max_num_per_bin:
            # random sample max_num_per_bin indices
            indices += random.sample(v, max_num_per_bin)
        elif len(v) <= min_num_per_bin:
            continue
        else:
            indices += v
    return indices


def filter_trajectory(
    trajs,
    cost_min=-np.inf,
    cost_max=np.inf,
    rew_min=-np.inf,
    rew_max=np.inf,
    cost_bins=60,
    rew_bins=50,
    max_num_per_bin=10,
    min_num_per_bin=1,
):
    returns = [compute_returns(traj) for traj in trajs]
    cost_returns = [compute_returns(traj) for traj in trajs]
    indices = grid_filter(
        cost_returns,
        returns,
        xmin=cost_min,
        xmax=cost_max,
        ymin=rew_min,
        ymax=rew_max,
        xbins=cost_bins,
        ybins=rew_bins,
        max_num_per_bin=max_num_per_bin,
        min_num_per_bin=min_num_per_bin,
    )
    trajs2 = [trajs[i] for i in indices]
    return trajs2


def get_nearest_point(
    original_data: np.ndarray,
    sampled_data: np.ndarray,
    max_rew_decrease: float = 1,
    beta: float = 1,
):
    """
    Given two arrays of data, finds the indices of the original data that are closest
    to each sample in the sampled data, and returns a list of those indices.

    Args:
        original_data: A 2D numpy array of the original data.
        sampled_data: A 2D numpy array of the sampled data.
        max_rew_decrease: A float representing the maximum reward decrease allowed.
        beta: A float used in calculating the distance between points.

    Returns:
        A list of integers representing the indices of the original data that are closest
        to each sample in the sampled data.
    """
    idxes = []
    original_idx = np.arange(0, original_data.shape[0])
    # for i in trange(sampled_data.shape[0], desc="Calculating nearest point"):
    for i in range(sampled_data.shape[0]):
        p = sampled_data[i, :]
        mask = original_data[:, 0] <= p[0]
        # mask = np.logical_and(original_data[:, 0] <= p[0], original_data[:, 0] >= p[0] - 5)
        delta = original_data[mask, :] - p
        dist = np.hypot(delta[:, 0], delta[:, 1])
        idx = np.argmin(dist)
        idxes.append(original_idx[mask][idx])
    counts = dict(Counter(idxes))

    new_idxes = []
    dist_fun = lambda x: 1 / (x + beta)
    for idx, num in counts.items():
        new_idxes.append(idx)
        if num > 1:
            p = original_data[idx, :]
            mask = original_data[:, 0] <= p[0]

            # the associated data should be: 1) smaller than the current cost 2) greater than certain reward
            mask = np.logical_and(
                original_data[:, 0] <= p[0],
                original_data[:, 1] >= p[1] - max_rew_decrease,
            )
            delta = original_data[mask, :] - p
            dist = np.hypot(delta[:, 0], delta[:, 1])
            dist = dist_fun(dist)
            sample_idx = np.random.choice(
                dist.shape[0], size=num - 1, p=dist / np.sum(dist)
            )
            new_idxes.extend(original_idx[mask][sample_idx.tolist()])
    return new_idxes


def augmentation(
    trajs: list,
    deg: int = 3,
    max_rew_decrease: float = 1,
    beta: float = 1,
    augment_percent: float = 0.3,
    max_reward: float = 1000.0,
    min_reward: float = 0.0,
):
    """
    Applies data augmentation to a list of trajectories,
    returning the augmented trajectories along with their indices
    and the Pareto frontier of the original data.
    Args:
        trajs: A list of dictionaries representing the original trajectories.
        deg: The degree of the polynomial used to fit the Pareto frontier.
        max_rew_decrease: The maximum amount by which the reward of an augmented trajectory can decrease compared to the original.
        beta: The scaling factor used to weigh the distance between cost and reward when finding nearest neighbors.
        augment_percent: The percentage of original trajectories to use for augmentation.
        max_reward: The maximum reward value for augmented trajectories.
        min_reward: The minimum reward value for augmented trajectories.

    Returns:
        nearest_idx: A list of indices of the original trajectories that are nearest to each augmented trajectory.
        aug_trajs: A list of dictionaries representing the augmented trajectories.
        pareto_frontier: A polynomial function representing the Pareto frontier of the original data.
    """
    reward_returns = [compute_returns(traj) for traj in trajs]
    cost_returns = [compute_cost_returns(traj) for traj in trajs]
    reward_returns = np.array(reward_returns, dtype=np.float64)
    cost_returns = np.array(cost_returns, dtype=np.float64)

    pareto = oapackage.ParetoDoubleLong()
    for i in range(reward_returns.shape[0]):
        w = oapackage.doubleVector((-cost_returns[i], reward_returns[i]))
        pareto.addvalue(w, i)

    # print pareto number
    pareto.show(verbose=1)
    pareto_idx = list(pareto.allindices())

    cost_returns_pareto = cost_returns[pareto_idx]
    reward_returns_pareto = reward_returns[pareto_idx]
    pareto_frontier = np.poly1d(
        np.polyfit(cost_returns_pareto, reward_returns_pareto, deg=deg)
    )

    sample_num = int(augment_percent * cost_returns.shape[0])
    # the augmented data should be within the cost return range of the dataset
    cost_returns_range = np.linspace(
        np.min(cost_returns), np.max(cost_returns), sample_num
    )
    pf_reward_returns = pareto_frontier(cost_returns_range)
    max_reward = max_reward * np.ones(pf_reward_returns.shape)
    min_reward = min_reward * np.ones(pf_reward_returns.shape)
    # sample the rewards that are above the pf curve and within the max_reward
    sampled_reward_returns = np.random.uniform(
        low=pf_reward_returns + min_reward, high=max_reward, size=sample_num
    )

    # associate each sampled (cost, reward) pair with a trajectory index
    original_data = np.hstack([cost_returns[:, None], reward_returns[:, None]])
    sampled_data = np.hstack(
        [cost_returns_range[:, None], sampled_reward_returns[:, None]]
    )
    nearest_idx = get_nearest_point(original_data, sampled_data, max_rew_decrease, beta)

    # relabel the dataset
    aug_trajs = []
    for i, target in zip(nearest_idx, sampled_data):
        target_cost_returns, target_reward_returns = target[0], target[1]
        associated_traj = copy.deepcopy(trajs[i])
        cost_returns = compute_discounted_cumsum_cost_returns(associated_traj, 1)
        reward_returns = compute_discounted_cumsum_returns(associated_traj, 1)
        cost_returns += target_cost_returns - cost_returns[0]
        reward_returns += target_reward_returns - reward_returns[0]
        aug_trajs.append(associated_traj)
    print(
        f"original data: {len(trajs)}, augment data: {len(aug_trajs)}, total: {len(trajs)+len(aug_trajs)}"
    )
    return trajs + aug_trajs


def random_augmentation(
    trajs: list,
    augment_percent: float = 0.3,
    aug_rmin: float = 0,
    aug_rmax: float = 600,
    aug_cmin: float = 5,
    aug_cmax: float = 50,
    cgap: float = 5,
    rstd: float = 1,
    cstd: float = 0.25,
):
    """
    Augments a list of trajectories with random noise.

    Args:
        trajs (list): A list of dictionaries, where each dictionary represents a trajectory
            and contains "returns" and "cost_returns" keys that hold the returns and cost returns
            for each time step of the trajectory.
        augment_percent (float, optional): The percentage of trajectories to augment.
        aug_rmin (float, optional): The minimum value for the augmented returns.
        aug_rmax (float, optional): The maximum value for the augmented returns.
        aug_cmin (float, optional): The minimum value for the augmented cost returns.
        aug_cmax (float, optional): The maximum value for the augmented cost returns.
        cgap (float, optional): The minimum distance between the augmented cost returns
        rstd (float, optional): The standard deviation of the noise to add to the returns.
        cstd (float, optional): The standard deviation of the noise to add to the cost returns.

    Returns:
        Tuple[List[int], List[Dict]]: A tuple containing two lists. The first list contains
            the indices of the original trajectories that were augmented. The second list contains
            the augmented trajectories, represented as dictionaries with "returns" and "cost_returns"
            keys.
    """
    reward_returns = [compute_returns(traj) for traj in trajs]
    cost_returns = [compute_cost_returns(traj) for traj in trajs]
    reward_returns = np.array(reward_returns, dtype=np.float64)
    cost_returns = np.array(cost_returns, dtype=np.float64)

    cmin = np.min(cost_returns)

    num = int(augment_percent * cost_returns.shape[0])
    sampled_cr = np.random.uniform(
        low=(aug_cmin, aug_rmin), high=(aug_cmax, aug_rmax), size=(num, 2)
    )

    idxes = []
    original_data = np.hstack([cost_returns[:, None], reward_returns[:, None]])
    original_idx = np.arange(0, original_data.shape[0])
    # for i in trange(sampled_data.shape[0], desc="Calculating nearest point"):
    for i in range(sampled_cr.shape[0]):
        p = sampled_cr[i, :]
        boundary = max(p[0] - cgap, cmin + 1)
        mask = original_data[:, 0] <= boundary
        # mask = np.logical_and(original_data[:, 0] <= p[0], original_data[:, 0] >= p[0] - 5)
        delta = original_data[mask, :] - p
        dist = np.hypot(delta[:, 0], delta[:, 1])
        idx = np.argmin(dist)
        idxes.append(original_idx[mask][idx])

    # relabel the dataset
    aug_trajs = []
    for i, target in zip(idxes, sampled_cr):
        target_cost_returns, target_reward_returns = target[0], target[1]
        associated_traj = copy.deepcopy(trajs[i])
        cost_returns = compute_discounted_cumsum_cost_returns(associated_traj, 1)
        reward_returns = compute_discounted_cumsum_returns(associated_traj, 1)
        cost_returns += (
            target_cost_returns
            - cost_returns[0]
            + np.random.normal(loc=0, scale=cstd, size=cost_returns.shape)
        )
        reward_returns += (
            target_reward_returns
            - reward_returns[0]
            + np.random.normal(loc=0, scale=rstd, size=reward_returns.shape)
        )
        aug_trajs.append(associated_traj)
    print(
        f"original data: {len(trajs)}, augment data: {len(aug_trajs)}, total: {len(trajs)+len(aug_trajs)}"
    )
    return trajs + aug_trajs


def select_optimal_trajectory(trajs: list, rmin:float = 0, cost_bins:float = 60, max_num_per_bin:int =1):
    """
    Selects the optimal trajectories from a list of trajectories based on their returns and costs.

    Args:
        trajs (list): A list of dictionaries, where each dictionary represents a trajectory and contains
                      the keys "returns" and "cost_returns".
        rmin (float): The minimum return that a trajectory must have in order to be considered optimal.
        cost_bins (int): The number of bins to divide the cost range into.
        max_num_per_bin (int): The maximum number of trajectories to select from each cost bin.

    Returns:
        list: A list of dictionaries representing the optimal trajectories.
    """
    reward_returns = [compute_returns(traj) for traj in trajs]
    cost_returns = [compute_cost_returns(traj) for traj in trajs]

    xmin, xmax = min(cost_returns), max(cost_returns)
    xbin_step = (xmax - xmin) / cost_bins
    # the key is x y bin index, the value is a list of indices
    bin_hashmap = defaultdict(list)
    for i in range(len(cost_returns)):
        if reward_returns[i] < rmin:
            continue
        x_bin_idx = (cost_returns[i] - xmin) // xbin_step
        bin_hashmap[x_bin_idx].append(i)

    # start filtering
    def sort_index(idx):
        return reward_returns[idx]

    indices = []
    for v in bin_hashmap.values():
        idx = heapq.nlargest(max_num_per_bin, v, key=sort_index)
        indices += idx

    traj2 = [trajs[i] for i in indices]
    return traj2


def data_augmentation(trajs: list, augmentation_method, augment_percent: float = 0.3,
    deg: int = 3, max_rew_decrease: float = 1, beta: float = 1, max_reward: float = 1000.0, min_reward: float = 0.0, 
    aug_rmin: float = 0, aug_rmax: float = 600, aug_cmin: float = 5, aug_cmax: float = 50, cgap: float = 5, rstd: float = 1, cstd: float = 0.25,
    rmin:float = 0, cost_bins:float = 60, max_num_per_bin:int =1):
    if augmentation_method == "Augmentation":
        return augmentation(trajs=trajs, deg=deg, max_rew_decrease=max_rew_decrease, beta=beta, augment_percent=augment_percent, max_reward=max_reward, min_reward=min_reward)
    elif augmentation_method == "RandomAugmentation":
        return random_augmentation(trajs=trajs, augment_percent=augment_percent, aug_rmin=aug_rmin, aug_rmax=aug_rmax, aug_cmin=aug_cmin, aug_cmax=aug_cmax, cgap=cgap, rstd=rstd, cstd=cstd)
    elif augmentation_method == "ParetoFrontierOnly":
        return select_optimal_trajectory(trajs=trajs, rmin=rmin, cost_bins=cost_bins, max_num_per_bin=max_num_per_bin)
    elif augmentation_method == "":
        return trajs
    else:
        raise NotImplementedError
