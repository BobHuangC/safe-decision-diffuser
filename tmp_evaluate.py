import argparse
import importlib
import json
import os

import orbax
from ml_collections import ConfigDict

from utilities.utils import dot_key_dict_to_nested_dicts
import gc
from utilities.utils import set_random_seed, str_to_list, to_arch

import csv

# carcircle
log_dir = "logs/tcdbc_dsrl/OfflineCarCircle-v0/transformer-gw_1.2-cdp_0.2-CDFNormalizer-normret_True/100/2024-3-13-4"
epochs = [1999]


def main():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", type=int, default=0)
    parser.add_argument("--evaluator_class", type=str, default="OnlineEvaluator")
    parser.add_argument("--num_eval_envs", type=int, default=10)
    parser.add_argument("--eval_n_trajs", type=int, default=20)
    parser.add_argument("--eval_env_seed", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    args = parser.parse_args()
    args.log_dir = log_dir
    args.epochs = epochs
    if args.g < 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.g)

    with open(os.path.join(args.log_dir, "variant.json"), "r") as f:
        variant = json.load(f)

    config = dot_key_dict_to_nested_dicts(variant)
    config = ConfigDict(config)


    # here specific set the trajs to 1 to evaluate the performance of the model
    config.eval_n_trajs = 1
    config.eval_env_seed = args.eval_env_seed
    config.eval_batch_size = args.eval_batch_size
    config.mode = "eval"

    evaluator = getattr(importlib.import_module("diffuser.trainer"), config.trainer)(
        config, use_absl=False
    )
    evaluator._setup()

    # eval_pro_data_record = {
    #     "epoch": [],
    #     "target_returns": [],
    # }



    eval_pro_data_record = {}


    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    target = {"agent_states": evaluator._agent.train_states}
    for epoch in args.epochs:
        ckpt_path = os.path.join(args.log_dir, f"checkpoints/model_{epoch}")
        print(f"ckpt_path: {ckpt_path}")
        restored = orbax_checkpointer.restore(ckpt_path, item=target)
        eval_params = {
            key: restored["agent_states"][key].params_ema
            or restored["agent_states"][key].params
            for key in evaluator._agent.model_keys
        }
        evaluator._evaluator.update_params(eval_params)


        for j in range(10):
            metrics = evaluator._evaluator.evaluate(epoch)

            for tmp_key in metrics.keys():
                if tmp_key not in eval_pro_data_record:
                    eval_pro_data_record[tmp_key] = [metrics[tmp_key]]
                else:
                    eval_pro_data_record[tmp_key].append(metrics[tmp_key])

            print(f"\033[92m Epoch {epoch}: {metrics} \033[00m\n")
        # eval_pro_data_record["epoch"].append(epoch)
        # eval_pro_data_record["target_returns"].append(eval_target_returns)

        gc.collect()

    import pandas as pd

    df = pd.DataFrame(eval_pro_data_record)
    os.makedirs(name=args.log_dir + "/eval", exist_ok=True)
    df.to_csv(
        f"{args.log_dir}/eval/tcr-{config.eval_target_cost_returns_list}--trr-{config.eval_target_reward_returns_list}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
