import argparse
import importlib
import absl
from utilities.utils import define_flags_with_default
import os
import sys
import orbax
from flax.training import orbax_utils

# test_model_path = "/NAS2020/Workspaces/DRLGroup/bohuang/bocode/safe-decision-diffuser/logs/cdbc_dsrl/OfflineAntRun-v0/tgt_90.0,1.0-guidew_2.0/300/checkpoints/model_0"
test_model_path = "/NAS2020/Workspaces/DRLGroup/bohuang/bocode/safe-decision-diffuser/logs/cdbc_dsrl/OfflineAntRun-v0/tgt_90.0,1.0-guidew_1.75/300/checkpoints/model_600"


def load_orbax_checkpoint(model_data_path: str):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data = orbax_checkpointer.restore(model_data_path)
    print(data)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-g", type=int, default=0)
    args, unknown_flags = parser.parse_known_args()
    if args.g < 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.g)

    from utilities.utils import import_file

    config = getattr(import_file(args.config, "default_config"), "get_config")()
    config = define_flags_with_default(**config)
    absl.flags.FLAGS(sys.argv[:1] + unknown_flags)

    trainer = getattr(
        importlib.import_module("diffuser.trainer"), absl.flags.FLAGS.trainer
    )(config)

    # 应该在这里进行参数的恢复, 然后直接测试应该就可以了
    data_restore = load_orbax_checkpoint(test_model_path)
    trainer.evaluate(data_restore["agent_states"])


if __name__ == "__main__":
    main()
