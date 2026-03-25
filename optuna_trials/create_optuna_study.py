

import optuna
from optuna.integration.skopt import SkoptSampler
from optuna.storages import RDBStorage
import argparse
from optuna.pruners import NopPruner
from optuna.samplers import RandomSampler

parser = argparse.ArgumentParser()
parser.add_argument("--study-name", type=str, required=True)
parser.add_argument("--storage", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

sampler = RandomSampler(seed=args.seed)
pruner = NopPruner()

study = optuna.create_study(
    study_name=args.study_name,
    storage=args.storage,
    sampler=sampler,
    pruner=pruner,
    direction="minimize",
    load_if_exists=False
)
