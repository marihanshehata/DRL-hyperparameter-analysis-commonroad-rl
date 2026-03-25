import subprocess
import os

ALGOS = ["ppo2", "sac"]   
SEEDS = [0, 42, 123, 999, 2024]


BASE_DB = "postgresql://optuna:pass@localhost:5432"

def create_database(db_name):
    subprocess.run([
        "psql",
        "-U", "optuna",
        "-c", f"CREATE DATABASE {db_name};"
    ])

for algo in ALGOS:
    for seed in SEEDS:


        db_name = f"{algo}_seed_{seed}"

	# create DB
	create_database(db_name)

	# use same name everywhere
	study_name = db_name
	storage = f"{BASE_DB}/{db_name}"

        print(f"\n=== {algo.upper()} | Seed {seed} ===")

        # Step 1: Create study
        subprocess.run([
            "python", "create_optuna_study.py",
            f"--study-name={study_name}",
            f"--storage={storage}",
            f"--seed={seed}"
        ])

        # Step 2: Run training
        env = {
            "OPTUNA_STORAGE": storage,
            "OPTUNA_STUDY_NAME": study_name
        }

        subprocess.run(
            [
                "python", "train_model.py",
                "--env=commonroad-v1",
                f"--algo={algo}",   
                "--save-freq=-1",
                "--optimize-hyperparams",
                "--n-trials=100",
                "--sampler=random",
                "--pruner=none",
                "--n_envs=8",
                "-n=100000",
                "--n-jobs=1",
                "--eval-freq=10000",
                f"--seed={seed}"
            ],
            env={**os.environ, **env} 
        )
