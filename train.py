import os
import sys
import absl
import argparse
import importlib

from utilities.utils import define_flags_with_default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, unknown_flags = parser.parse_known_args()

    from utilities.utils import import_file

    config = getattr(import_file(args.config, "default_config"), "get_config")()
    config = define_flags_with_default(**config)
    absl.flags.FLAGS(sys.argv[:1] + unknown_flags)

    trainer = getattr(importlib.import_module("diffuser.trainer"), absl.flags.FLAGS.trainer)(config)
    trainer.train()
    os._exit(os.EX_OK)


if __name__ == "__main__":
    main()
