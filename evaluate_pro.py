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

# ant run
log_dir = "logs/cdbc_dsrl/OfflineAntRun-v0/tgt_700.0,10, 750.0,20, 800.0,40-guidew_2.0/300/2_16_3"
epochs = [650, 700, 1200, 1550, 3150]

# ant run1
# log_dir = "logs/test-cdbc_dsrl/OfflineAntRun-v0/tgt_650.0, 0, 700.0,10, 750.0,20, 800.0,40-guidew_2.0/300/2_23_1"
# epochs = [400, 575, 625, 1175, 1275, 1400, 1600]


# car circle

# car run
# log_dir = "logs/cdbc_dsrl/OfflineCarRun-v0/tgt_575.0,10, 575.0,20, 575.0,40-guidew_2.0/300/2_17_3"
# epochs = [250, 2300, 2400, 2950, 3700]

# drone circle
# log_dir = "logs/cdbc_dsrl/OfflineDroneCircle-v0/tgt_700.0,10, 750.0,20, 800.0,40-guidew_2.0/300/2_17_2"
# epochs = [2550, 3000, 3650, 3700, 3750, 3800, 3850, 3900, 3950, 4000]

# drone run
# log_dir = "logs/cdbc_dsrl/OfflineDroneRun-v0/tgt_400.0,10, 500.0,20, 600.0,40-guidew_2.0/300/2_17_2"
# epochs = [650, 750, 1000, 1100, 1250, 1450, 1700, 2000, 2200]


def main():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    parser = argparse.ArgumentParser()
    # parser.add_argument("log_dir", type=str)
    parser.add_argument("-g", type=int, default=0)
    parser.add_argument("--evaluator_class", type=str, default="OnlineEvaluator")
    parser.add_argument("--num_eval_envs", type=int, default=10)
    parser.add_argument("--eval_n_trajs", type=int, default=20)
    parser.add_argument("--eval_env_seed", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    # parser.add_argument("--epochs", type=int, nargs="+", required=True)
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

    # rewrite configs
    config.evaluator_class = args.evaluator_class
    config.num_eval_envs = args.num_eval_envs
    config.eval_n_trajs = args.eval_n_trajs
    config.eval_env_seed = args.eval_env_seed
    config.eval_batch_size = args.eval_batch_size

    # config.returns_condition = True
    # config.cost_returns_condition = True
    config.mode = "eval"

    # 1, 3, 5, 7
    # config.condition_guidance_w = 1.7
    config.target_returns = "200.0, 0.0"

    # the original is 0.5
    # config.algo_cfg.sample_temperature = 0.9

    evaluator = getattr(importlib.import_module("diffuser.trainer"), config.trainer)(
        config, use_absl=False
    )
    evaluator._setup()

    eval_pro_data_record = {
        "epoch": [],
        "target_returns": [],
    }

    eval_target_returns = ""
    target_reward_returns_list = str_to_list(config.eval_target_reward_returns_list)
    target_cost_returns_list = str_to_list(config.eval_target_cost_returns_list)

    for reward in target_reward_returns_list:
        for cost in target_cost_returns_list:
            eval_target_returns += f"{reward}, {cost},"
    eval_target_returns = eval_target_returns[:-1]

    evaluator._reset_target_returns(eval_target_returns)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    target = {"agent_states": evaluator._agent.train_states}
    for epoch in args.epochs:
        ckpt_path = os.path.join(args.log_dir, f"checkpoints/model_{epoch}")
        restored = orbax_checkpointer.restore(ckpt_path, item=target)
        eval_params = {
            key: restored["agent_states"][key].params_ema
            or restored["agent_states"][key].params
            for key in evaluator._agent.model_keys
        }
        evaluator._evaluator.update_params(eval_params)
        metrics = evaluator._evaluator.evaluate(epoch)

        for tmp_key in metrics.keys():
            if tmp_key not in eval_pro_data_record:
                eval_pro_data_record[tmp_key] = [metrics[tmp_key]]
            else:
                eval_pro_data_record[tmp_key].append(metrics[tmp_key])

        print(f"\033[92m Epoch {epoch}: {metrics} \033[00m\n")
        eval_pro_data_record["epoch"].append(epoch)
        eval_pro_data_record["target_returns"].append(eval_target_returns)

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
