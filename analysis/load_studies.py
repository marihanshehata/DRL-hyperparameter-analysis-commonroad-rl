import optuna
from configs import ALGOS, SEEDS, BASE_DB

def load_all_studies():
    studies = {}

    for algo in ALGOS:
        studies[algo] = {}

        for seed in SEEDS:
            db_name = f"{algo}_seed_{seed}"
            storage = f"{BASE_DB}/{db_name}"
            study_name = db_name

            print(f"Loading {study_name}")
      
            study = optuna.load_study(
                study_name=study_name,
                storage=storage
            )

            studies[algo][seed] = study

    return studies
