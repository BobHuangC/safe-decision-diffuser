import argparse
import importlib
import json
import os

import orbax
from ml_collections import ConfigDict

from utilities.utils import dot_key_dict_to_nested_dicts


# CarCircle
# the checkpoint for test(already backup)
log_dir = "logs/tcdbc_dsrl/OfflineCarCircle-v0/transformer-gw_1.2-cdp_0.2-CDFNormalizer-normret_True/100/2024-3-15-1"
epoch = 1999

retrain_flag = "4-5-1"


def main():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", type=int, default=0)
    parser.add_argument("--evaluator_class", type=str, default="OnlineEvaluator")
    parser.add_argument("--num_eval_envs", type=int, default=10)
    parser.add_argument("--eval_n_trajs", type=int, default=20)
    parser.add_argument("--eval_env_seed", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    # parser.add_argument("--epochs", type=int, nargs="+", required=True)
    args = parser.parse_args()
    args.log_dir = log_dir
    args.epochs = epoch
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

    config.mode = 'retrain'
    config.restored_model = log_dir
    config.restored_epoch = epoch

    config.log_dir_format = config.log_dir_format + "/retrain/" + str(retrain_flag)



    train_config = config
    trainer = getattr(
        importlib.import_module("diffuser.trainer"), "TransformerCondDiffusionBCTrainer"
    )(config = train_config, use_absl = False)


    ckpt_path = os.path.join(args.log_dir, f"checkpoints/model_{epoch}")
    trainer.train(restored_ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
