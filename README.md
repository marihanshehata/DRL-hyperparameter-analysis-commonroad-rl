# DRL-hyperparameter-analysis-commonroad-rl

Hyperparameter sensitivity and regime analysis of PPO and SAC for autonomous driving using CommonRoad-RL.

---
## 📄 Publications

This repository supports the experiments and analysis presented in the following works:

### Published Paper

* "Hyperparameter Sensitivity Analysis of Deep Reinforcement Learning for Autonomous Driving"
  Available at: https://aircconline.com/csit/papers/vol15/csit152205.pdf

### Submitted Paper

* "Hyperparameter Regimes and Learning Dynamics in Deep
Reinforcement Learning for Autonomous Driving" *(under review)*

## Overview

This repository contains the code and analysis for studying the effect of hyperparameters on Deep Reinforcement Learning (DRL) algorithms in autonomous driving scenarios.

The experiments are conducted using the **CommonRoad-RL** framework and the **highD dataset** , with a focus on:

* PPO (Proximal Policy Optimization)
* SAC (Soft Actor-Critic)

Hyperparameter optimization is performed using **Optuna**, and results are analyzed to understand performance sensitivity across different configurations and random seeds.

---

## Project Structure

```
project/
│
├── README.md
├── analysis/
│   ├── script/       
│   └── results/          
│
├── optuna_trials/
│   ├── run_optuna.py     # Main script to run all Optuna studies
│   ├── configs.yaml      
│   ├── create_optuna_study.py
│   ├── train_model.py
│   └── utils_run/
│       ├── hyperparams_opt.py
│       └── callbacks.py
│
└── requirements.txt
```

---

## ⚙️ Optuna Trials Generation

Hyperparameter optimization is performed using Optuna to generate 5 Optuna studies for each algorithm with the following design:

* sampler: Random search
* Pruner: No pruner 
* Each Optuna study corresponds to a specific:
  * algorithm (PPO or SAC)
  * random seed
    
* Each experiment is stored in a **separate PostgreSQL database**
* Each database contains exactly one Optuna study
* each study contains a 100 Compelete Optuna trials
   

### Running Experiments

All studies can be reproduced using:

```bash
python optuna_trials/run_optuna.py
```

This script:

1. Creates a PostgreSQL database for each (algorithm, seed) pair
2. Initializes an Optuna study
3. Runs hyperparameter optimization via `train_model.py`

---

## Storage Design

Each Optuna study is stored in a **separate PostgreSQL database**.

### Naming Convention

```
{algorithm}_seed_{seed}
```

### Example

```
ppo2_seed_0
ppo2_seed_42
sac_seed_123
```

Each database contains a single study with the same name.

---

## Modifications to CommonRoad-RL

This work builds on the original CommonRoad-RL framework with the following modifications:

### 1. Modified Hyperparameter Search Space

* The hyperparameter ranges for PPO and SAC were adapted
* This enables a more comprehensive and controlled exploration of the search space

### 2. Logging Disabled

* Logging was removed or reduced to minimize overhead
* This improves runtime performance during large-scale Optuna studies

### 3. Optuna Storage Backend Changed

* Default storage (e.g., pickle/in-memory) was replaced with **PostgreSQL**
* This allows  persistent storage, scalability and safer parallel or repeated runs

### 4. Study Isolation via Databases

* Each study is stored in a **separate PostgreSQL database**


---

## Reproducibility

This project builds on:

CommonRoad-RL
highD dataset (for traffic scenarios)

Please install CommonRoad-RL separately and ensure it is accessible in your environment.

1. Install dependencies

Create a virtual environment (recommended), then install:

```bash
pip install -r requirements.txt
```

The experiments were developed using Python 3.7.12


2. Set up PostgreSQL:

Install and run PostgreSQL, then:

Create a user (e.g., optuna)
Ensure the user has permission to create databases

The connection string format used in this project is:

```
postgresql://user:password@localhost:5432/
```
Each Optuna study will automatically create and use a separate database.

3. Run:

```bash
python optuna_trials/run_optuna.py
```

---

## Analysis

All analysis is located in:

```
analysis/script/
```

This includes:

* hyperparameter sensitivity analysis fANOVA (functional ANOVA)
* visualization of optimization results
* Clustering of hyperparameter configurations using K-Means
* plotting learning curves for prolonged training of choosen learning regiemes

---

## Notes

* The original CommonRoad-RL codebase is **not included** in this repository.
* Only the modified components required for the experiments are provided.
* Users must clone the original framework separately and integrate the provided modifications.

---

##  Purpose

This repository is designed to:

* study the effect of hyperparameters in DRL for autonomous driving
* provide reproducible Optuna-based experimentation
* support further research in hyperparameter optimization and RL robustness

---

