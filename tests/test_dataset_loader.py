import importlib
import sys

import absl
import torch

from utilities.data_utils import numpy_collate
from utilities.utils import define_flags_with_default, import_file


def load_datasets():
    config = getattr(
        import_file("configs/dql_walker_mdreplay.py", "default_config"), "get_config"
    )()
    config = define_flags_with_default(**config)
    absl.flags.FLAGS(sys.argv[:1])

    trainer = getattr(
        importlib.import_module("diffuser.trainer"), absl.flags.FLAGS.trainer
    )(config)

    dataset, _ = trainer._setup_dataset()
    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=32,
        collate_fn=numpy_collate,
        num_workers=32,
        pin_memory=True,
    )
    data = next(iter(dataloader))


if __name__ == "__main__":
    load_datasets()
