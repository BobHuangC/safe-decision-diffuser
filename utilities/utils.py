# Copyright 2023 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General utils for training."""

import functools
import importlib
import os
import pprint
import random
import string
import sys
import tempfile
import time
import uuid
from copy import copy
from socket import gethostname

import absl.flags
import cloudpickle as pickle
import numpy as np
import wandb
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags

from utilities.jax_utils import init_rng


def to_arch(string):
    return tuple(int(x) for x in string.split("-"))


def str_to_list(string):
    return [float(x) for x in string.split(",")]


def apply_conditioning(x, conditions, condition_dim: int):
    for t, val in conditions.items():
        assert condition_dim is not None
        x = x.at[:, t[0] : t[1], :condition_dim].set(val)
    return x


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def dot_key_dict_to_nested_dicts(dict_in):
    dict_out = {}
    for key, value in dict_in.items():
        cur = dict_out
        *keys, leaf = key.split(".")
        for k in keys:
            cur = cur.setdefault(k, {})
        cur[leaf] = value
    return dict_out


class Timer(object):
    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        # config.team = "safediff"
        config.online = True
        # config.online = False
        config.output_dir = "logs"
        config.random_delay = 0.0
        config.log_dir = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant):
        self.config = self.get_default_config(config)

        if self.config.log_dir is None:
            self.config.log_dir = uuid.uuid4().hex

        if self.config.output_dir == "":
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(
                self.config.output_dir, self.config.log_dir
            )
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        self.run = wandb.init(
            reinit=True,
            config=self._variant,
            project=self.config.log_dir.split("/")[1],
            dir=self.config.output_dir,
            name=self.config.log_dir,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode="online" if self.config.online else "offline",
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        with open(os.path.join(self.config.output_dir, filename), "wb") as fout:
            pickle.dump(obj, fout)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, "automatically defined flag")
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, "automatically defined flag")
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, "automatically defined flag")
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, "automatically defined flag")
        else:
            raise ValueError("Incorrect value type")
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)


def print_flags(flags, flags_def):
    logging.info(
        "Running training with hyperparameters: \n{}".format(
            pprint.pformat(
                [
                    "{}: {}".format(key, val)
                    for key, val in get_user_flags(flags, flags_def).items()
                ]
            )
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                output["{}.{}".format(prefix, key)] = val
            else:
                output[key] = val
    return output


def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}


def import_file(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class DotFormatter(string.Formatter):
    def get_field(self, field_name, args, kwargs):
        return (self.get_value(field_name, args, kwargs), field_name)


# one way to choose the best epoch
# use the epoch with the minimum policy gradient norm as the best epoch
def epoch_selection(log_csv_path, save_epoch):
    import pandas as pd

    # Read the progress.csv file
    df = pd.read_csv(log_csv_path)
    candidate_epochs = df["epoch"].unique()
    candidate_epochs = [i for i in candidate_epochs if i % save_epoch == 0]
    candidate_epochs.pop(len(candidate_epochs) - 1)
    candidate_metric = []
    for epoch in candidate_epochs:
        _grad_value = df.iloc[epoch]["agent/policy_grad_norm"]
        for _ in range(1, save_epoch):
            _grad_value += df.iloc[epoch + _]["agent/policy_grad_norm"]
            _grad_value += df.iloc[epoch - _]["agent/policy_grad_norm"]
        _grad_value /= 1 + 2 * (save_epoch - 1)

        candidate_metric.append(float(_grad_value))

    candidate_final = min(candidate_metric)
    candidate_epoch = candidate_epochs[candidate_metric.index(candidate_final)]
    return candidate_epoch
