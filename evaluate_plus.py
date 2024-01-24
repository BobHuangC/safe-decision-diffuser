import argparse
import importlib
import json
import os

import orbax
from ml_collections import ConfigDict

from utilities.utils import dot_key_dict_to_nested_dicts

import pandas as pd
import gc


log_dir = "logs/eval-cdbc_dsrl/OfflineAntRun-v0/tgt_390.0,1.0-guidew_1.75/300"
epochs = [3250]

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# evaluate the model with the following target returns to evaluate the model properly
target_reward_returns_list = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0]
# target_reward_returns_list = [50.0, 100.0, 150.0, 200.0, ]
# target_reward_returns_list = [250.0, 300.0, 350.0, 400.0 ]
# target_reward_returns_list = [450.0, 500.0, 550.0, ]
target_reward_returns_list = [600.0, 650.0, 700.0 ]

target_cost_returns_list = [0.0, 2.0, 5.0, 7.0, 10.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0,]


eval_plus_data_record = {"epoch":[], "target_reward_return":[], "target_cost_return":[], "average_reward_return":[], 
"average_cost_return":[], "reward_return":[], "cost_return":[]}

# target_reward_returns_list = [500.0]
# target_cost_returns_list = [50.0, ]

target_returns_list = []
for reward in target_reward_returns_list:
    for cost in target_cost_returns_list:
        target_returns_list.append(f"{reward}, {cost}")



target_returns_list_cost_none = []
target_returns_list_reward_none = []

# cost none 对应的是cost为none, 我们默认设置为cost=6.6表示为none, 但是在实际的evaluate中不会传入cost, 所以这里的cost=6.6是没有意义的
for reward in target_reward_returns_list:
    target_returns_list_cost_none.append(f"{reward}, 6.6")


# reward none 对应的是reward为none, 我们默认设置为reward=666表示为none, 但是在实际的evaluate中不会传入reward, 所以这里的reward=666是没有意义的
for cost in target_cost_returns_list:
    target_returns_list_reward_none.append(f"666, {cost}")


def main():
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

    config.returns_condition = True
    config.cost_returns_condition = True
    print(config.env_ts_condition, ' this is the env_ts_condition of config')
    print(config.returns_condition, ' this is the returns_condition of config')
    print(config.cost_returns_condition, ' this is the cost_returns_condition of config')

    config.condition_guidance_w = 2.0
    # config.target_returns = target_returns_list[0]
    config.target_returns = "200.0, 0.0"
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


    # # set the target reward returns to be None
    # config.returns_condition = False
    # config.cost_returns_condition = True
    # config.target_returns = "666, 6.6"

    # trainer = getattr(importlib.import_module("diffuser.trainer"), config.trainer)(
    #     config, use_absl=False
    # )
    # trainer._setup()

    # for tmp_target_returns in target_returns_list_reward_none:
    #     trainer._reset_target_returns(tmp_target_returns)
    #     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    #     target = {"agent_states": trainer._agent.train_states}
    #     for epoch in args.epochs:
    #         ckpt_path = os.path.join(args.log_dir, f"checkpoints/model_{epoch}")
    #         restored = orbax_checkpointer.restore(ckpt_path, item=target)
    #         eval_params = {
    #             key: restored["agent_states"][key].params_ema
    #             or restored["agent_states"][key].params
    #             for key in trainer._agent.model_keys
    #         }
    #         trainer._evaluator.update_params(eval_params)
    #         metrics = trainer._evaluator.evaluate(epoch)
    #         print(f"\033[92m Epoch {epoch}: {metrics} \033[00m\n")
    #         eval_plus_data_record["epoch"].append(epoch)
    #         eval_plus_data_record["target_reward_return"].append("None")
    #         eval_plus_data_record["target_cost_return"].append(tmp_target_returns.split(",")[1])
    #         eval_plus_data_record["average_reward_return"].append(metrics["average_return"])
    #         eval_plus_data_record["average_cost_return"].append(metrics["average_cost_return"])
    #         eval_plus_data_record["reward_return"].append(metrics["return_record"])
    #         eval_plus_data_record["cost_return"].append(metrics["cost_return_record"])
    #         gc.collect()

    

    # # set the target cost returns to be None
    # config.returns_condition = True
    # config.cost_returns_condition = False
    # config.target_returns = "666, 6.6"

    # trainer = getattr(importlib.import_module("diffuser.trainer"), config.trainer)(
    #     config, use_absl=False
    # )
    # trainer._setup()

    # for tmp_target_returns in target_returns_list_cost_none:
    #     trainer._reset_target_returns(tmp_target_returns)
    #     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    #     target = {"agent_states": trainer._agent.train_states}
    #     for epoch in args.epochs:
    #         ckpt_path = os.path.join(args.log_dir, f"checkpoints/model_{epoch}")
    #         restored = orbax_checkpointer.restore(ckpt_path, item=target)
    #         eval_params = {
    #             key: restored["agent_states"][key].params_ema
    #             or restored["agent_states"][key].params
    #             for key in trainer._agent.model_keys
    #         }
    #         trainer._evaluator.update_params(eval_params)
    #         metrics = trainer._evaluator.evaluate(epoch)
    #         print(f"\033[92m Epoch {epoch}: {metrics} \033[00m\n")
    #         eval_plus_data_record["epoch"].append(epoch)
    #         eval_plus_data_record["target_reward_return"].append(tmp_target_returns.split(",")[0])
    #         eval_plus_data_record["target_cost_return"].append("None")
    #         eval_plus_data_record["average_reward_return"].append(metrics["average_return"])
    #         eval_plus_data_record["average_cost_return"].append(metrics["average_cost_return"])
    #         eval_plus_data_record["reward_return"].append(metrics["return_record"])
    #         eval_plus_data_record["cost_return"].append(metrics["cost_return_record"])
    #         gc.collect()

    # config.returns_condition = False
    # config.cost_returns_condition = False
    # config.target_returns = "666, 6.6"

    # trainer = getattr(importlib.import_module("diffuser.trainer"), config.trainer)(
    #     config, use_absl=False
    # )
    # trainer._setup()

    # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # target = {"agent_states": trainer._agent.train_states}

    # for epoch in args.epochs:
    #     ckpt_path = os.path.join(args.log_dir, f"checkpoints/model_{epoch}")
    #     restored = orbax_checkpointer.restore(ckpt_path, item=target)
    #     eval_params = {
    #         key: restored["agent_states"][key].params_ema
    #         or restored["agent_states"][key].params
    #         for key in trainer._agent.model_keys
    #     }
    #     trainer._evaluator.update_params(eval_params)
    #     metrics = trainer._evaluator.evaluate(epoch)
    #     print(f"\033[92m Epoch {epoch}: {metrics} \033[00m\n")
    #     eval_plus_data_record["epoch"].append(epoch)
    #     eval_plus_data_record["target_reward_return"].append("None")
    #     eval_plus_data_record["target_cost_return"].append("None")
    #     eval_plus_data_record["average_reward_return"].append(metrics["average_return"])
    #     eval_plus_data_record["average_cost_return"].append(metrics["average_cost_return"])
    #     eval_plus_data_record["reward_return"].append(metrics["return_record"])
    #     eval_plus_data_record["cost_return"].append(metrics["cost_return_record"])
    #     gc.collect()





    # 创建一个DataFrame对象
    df = pd.DataFrame(eval_plus_data_record)

    # 将DataFrame保存为.csv文件
    os.makedirs(f'eval_plus_data/test-1-24/{str(args.log_dir).replace("/", "-")}', exist_ok=True)
    df.to_csv(f'eval_plus_data/test-1-24/{str(args.log_dir).replace("/", "-")}/trr-{str(target_reward_returns_list)}-tcr-{str(target_cost_returns_list)}-guid_w-{config.condition_guidance_w}.csv', index=False)

if __name__ == "__main__":
    main()
