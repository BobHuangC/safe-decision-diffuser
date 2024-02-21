from collections import deque
from typing import Any, Dict

import numpy as np

from .base_evaluator import BaseEvaluator


class OnlineEvaluator(BaseEvaluator):
    eval_mode: str = "online"

    def __init__(self, config, policy, eval_sampler):
        super().__init__(config, policy)
        self._eval_sampler = eval_sampler

        self._act_methods = self._cfgs.act_method.split("-")
        self._recent_returns = {
            method: deque(maxlen=10) for method in self._act_methods
        }
        self._best_returns = {method: -float("inf") for method in self._act_methods}
        self._recent_cost_returns = {
            method: deque(maxlen=10) for method in self._act_methods
        }

    def evaluate(self, epoch: int) -> Dict[str, Any]:
        metrics = {}
        for method in self._act_methods:
            ret_trajs = self._sample_trajs(method)
            target_nums = len(ret_trajs)
            # post: the flag representing the method
            post = "" if len(self._act_methods) == 1 else "_" + method

            for target_idx in range(target_nums):
                trajs = ret_trajs[target_idx]
                metrics[f"t{target_idx}-average_return" + post] = []
                metrics[f"t{target_idx}-return_std" + post] = []

                metrics[f"t{target_idx}-average_return" + post] = np.mean(
                    [np.sum(t["rewards"]) for t in trajs]
                )

                metrics[f"t{target_idx}-return_std" + post] = np.std(
                    [np.sum(t["rewards"]) for t in trajs]
                )

                metrics[f"t{target_idx}-average_cost_return" + post] = cur_cost = np.mean(
                    [np.sum(t["costs"]) for t in trajs]
                )

                metrics[f"t{target_idx}-cost_return_std" + post] = np.std(
                    [np.sum(t["costs"]) for t in trajs]
                )

                metrics[f"t{target_idx}-return_record" + post] = [np.sum(t["rewards"]) for t in trajs]

                metrics[f"t{target_idx}-cost_return_record" + post] = [np.sum(t["costs"]) for t in trajs]

                metrics[f"t{target_idx}-average_traj_length" + post] = np.mean(
                    [len(t["rewards"]) for t in trajs]
                )


                # TODO: the following to be rewritten(2024.2.21.16:53)
                if hasattr(self._eval_sampler.env, "set_target_cost"):
                    cur_return, cur_cost_return = np.mean(
                        [
                            self._eval_sampler.env.get_normalized_score(
                                np.sum(t["rewards"]), np.sum(t["costs"])
                            )
                            for t in trajs
                        ],
                        axis=0,
                    )
                    

                    metrics[f"t{target_idx}-normalized_return_record" + post] = [i[0] for i in [
                            self._eval_sampler.env.get_normalized_score(
                                np.sum(t["rewards"]), np.sum(t["costs"])
                            )
                            for t in trajs
                        ]]

                    metrics[f"t{target_idx}-normalized_cost_return_record" + post] = [i[1] for i in [
                            self._eval_sampler.env.get_normalized_score(
                                np.sum(t["rewards"]), np.sum(t["costs"])
                            )
                            for t in trajs
                        ]]
                else:
                    cur_return = np.mean(
                        [
                            self._eval_sampler.env.get_normalized_score(
                                np.sum(t["rewards"])
                            )
                            for t in trajs
                        ],
                    )

                    metrics[f"t{target_idx}-normalized_return_record" + post] = [
                            self._eval_sampler.env.get_normalized_score(
                                np.sum(t["rewards"])
                            )
                            for t in trajs
                        ]
                    cur_cost_return = cur_cost

                metrics[f"t{target_idx}-average_normalized_return" + post] = cur_return

                metrics[f"t{target_idx}-average_normalized_cost_return" + post] = cur_cost_return

                self._recent_returns[method].append(cur_return)
                self._recent_cost_returns[method].append(cur_cost)

                metrics[f"t{target_idx}-average_10_normalized_return" + post] = np.mean(
                    self._recent_returns[method]
                )

                metrics[f"t{target_idx}-best_normalized_return" + post] = self._best_returns[method] = max(
                    self._best_returns[method], cur_return
                )

                metrics[f"t{target_idx}-average_10_normalized_cost_return" + post] = np.mean(
                    self._recent_cost_returns[method]
                )

        self.dump_metrics(metrics, epoch, suffix="_online")
        return metrics

    def _sample_trajs(self, act_method: str):
        self._policy.act_method = act_method
        trajs = self._eval_sampler.sample(
            self._policy,
            self._cfgs.eval_n_trajs,
            deterministic=True,
        )
        return trajs
