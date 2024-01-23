import argparse
import importlib
import json
import os

import orbax
from ml_collections import ConfigDict

from utilities.utils import dot_key_dict_to_nested_dicts

import pandas as pd
import gc

eval_plus_data_record = {"epoch":[], "target_reward_return":[], "target_cost_return":[], "average_reward_return":[], 
"average_cost_return":[], "reward_return":[], "cost_return":[]}

# evaluate the model with the following target returns to evaluate the model properly
target_reward_returns_list = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0]
target_reward_returns_list = [50.0, 100.0, 150.0, 200.0,]
target_cost_returns_list = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

target_returns_list = []
for reward in target_reward_returns_list:
    for cost in target_cost_returns_list:
        target_returns_list.append(f"{reward}, {cost}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)
    parser.add_argument("-g", type=int, default=0)
    parser.add_argument("--evaluator_class", type=str, default="OnlineEvaluator")
    parser.add_argument("--num_eval_envs", type=int, default=10)
    parser.add_argument("--eval_n_trajs", type=int, default=20)
    parser.add_argument("--eval_env_seed", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    args = parser.parse_args()
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

    config.target_returns = target_returns_list[0]
    trainer = getattr(importlib.import_module("diffuser.trainer"), config.trainer)(
        config, use_absl=False
    )
    trainer._setup()


    for tmp_target_returns in target_returns_list:
        trainer._reset_target_returns(tmp_target_returns)
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        target = {"agent_states": trainer._agent.train_states}
        for epoch in args.epochs:
            ckpt_path = os.path.join(args.log_dir, f"checkpoints/model_{epoch}")
            restored = orbax_checkpointer.restore(ckpt_path, item=target)
            eval_params = {
                key: restored["agent_states"][key].params_ema
                or restored["agent_states"][key].params
                for key in trainer._agent.model_keys
            }
            trainer._evaluator.update_params(eval_params)
            metrics = trainer._evaluator.evaluate(epoch)
            print(f"\033[92m Epoch {epoch}: {metrics} \033[00m\n")
            eval_plus_data_record["epoch"].append(epoch)
            eval_plus_data_record["target_reward_return"].append(tmp_target_returns.split(",")[0])
            eval_plus_data_record["target_cost_return"].append(tmp_target_returns.split(",")[1])
            eval_plus_data_record["average_reward_return"].append(metrics["average_return"])
            eval_plus_data_record["average_cost_return"].append(metrics["average_cost_return"])
            eval_plus_data_record["reward_return"].append(metrics["return_record"])
            eval_plus_data_record["cost_return"].append(metrics["cost_return_record"])
            gc.collect()


    # 创建一个DataFrame对象
    df = pd.DataFrame(eval_plus_data_record)

    # 将DataFrame保存为.csv文件
    df.to_csv(f'eval_plus_data/test-{str(args.log_dir).replace("/", "-")}-{str(target_reward_returns_list)}.csv', index=False)

if __name__ == "__main__":
    main()
