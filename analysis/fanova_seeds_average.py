import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_rel
import matplotlib.ticker as mtick  # (put once at top of file)



# === Set this to where your seed folders are ===
sac_base_path = "logs/sac_sparse_cont/seeds"
ppo_base_path = "logs/ppo_sparse_cont/seeds"# <-- CHANGE THIS!

# List of seed subdirectories
sac_seed_dirs = [os.path.join(sac_base_path, d) for d in os.listdir(sac_base_path) if d.startswith("seed")]

ppo_seed_dirs = [os.path.join(ppo_base_path, d) for d in os.listdir(ppo_base_path) if d.startswith("seed")]

main_file = "fanova_main_effects.txt"
interaction_file = "fanova_interaction_effects.txt"

'''
def parse_main_effects(file_path):
    effects = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line and '(' not in line:
                param, val = line.strip().split(':')
                effects[param.strip()] = float(val.strip())
    return effects
'''
def parse_main_effects(file_path):
    effects = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line and '(' not in line:
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    param, val = parts
                    try:
                        effects[param.strip()] = float(val.strip())
                    except ValueError:
                        continue  # Skip malformed or empty values
    return effects


def parse_interaction_effects(file_path):
    effects = {}
    with open(file_path, 'r') as f:
        for line in f:
            if '(' in line and ':' in line:
                try:
                    key, val = line.strip().split(':', 1)
                    key = key.strip().replace('(', '').replace(')', '').replace(' ', '')
                    val = val.strip()
                    if val:  # Make sure value is not empty
                        effects[key] = float(val)
                except ValueError:
                    print(f"Skipping malformed line: {line.strip()}")
    return effects


def aggregate_effects(seed_dirs, filename, parser):
    data = defaultdict(list)
    for seed_dir in seed_dirs:
        file_path = os.path.join(seed_dir, filename)
        if os.path.exists(file_path):
            parsed = parser(file_path)
            for key, val in parsed.items():
                data[key].append(val)
        else:
            print(f"Warning: File not found: {file_path}")
    return data

def plot_error_bars(data, title, output_file, top_k=None):
    # Sort by mean importance descending
    sorted_items = sorted(data.items(), key=lambda x: np.mean(x[1]), reverse=True)

    # Optionally, keep only top_k items
    if top_k is not None:
        sorted_items = sorted_items[:top_k]

    labels = [k for k, _ in sorted_items]
    means = [np.mean(v) for _, v in sorted_items]
    stds = [np.std(v) for _, v in sorted_items]

    x = np.arange(len(labels))

    plt.figure(figsize=(max(10, len(labels) * 0.6), 6))  # auto widen based on number of bars
    bars = plt.bar(x, means, yerr=stds, capsize=5, width=0.7, alpha=0.8, color="#4B8BBE", edgecolor='black', linewidth=0.6)

    plt.xticks(x, labels, rotation=90, ha='center', fontsize=15)
    plt.ylabel("Importance", fontsize=15)
    plt.title(title, fontsize=17)


    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def save_averaged_effects(data, output_file, top_k=None):
    # Sort by mean importance descending
    sorted_items = sorted(data.items(), key=lambda x: np.mean(x[1]), reverse=True)

    if top_k is not None:
        sorted_items = sorted_items[:top_k]

    with open(output_file, 'w') as f:
        f.write("Effect,Mean,Std\n")
        for key, values in sorted_items:
            mean = np.mean(values)
            std = np.std(values)
            f.write(f"{key},{mean:.6f},{std:.6f}\n")

# === MAIN EFFECTS ===
ppo_main_data = aggregate_effects(ppo_seed_dirs, main_file, parse_main_effects)
plot_error_bars(ppo_main_data, " (a) FANOVA Main Effects for PPO (Mean ± Std across seeds)", "ppo_main_effects_avg.png")
save_averaged_effects(ppo_main_data, "ppo_main_effects_avg.csv")  # <--- save to CSV

# === INTERACTION EFFECTS ===
ppo_interaction_data = aggregate_effects(ppo_seed_dirs, interaction_file, parse_interaction_effects)
plot_error_bars(ppo_interaction_data, "(a) FANOVA Interaction Effects for PPO (Mean ± Std across seeds)", "ppo_interaction_effects_avg.png")
plot_error_bars(ppo_interaction_data, "(a) Top 20 FANOVA Interaction Effects for PPO (Mean ± Std across seeds)", "ppo_interaction_effects_top20.png", top_k=20)
save_averaged_effects(ppo_interaction_data, "ppo_interaction_effects_avg.csv")  # <--- save to CSV
save_averaged_effects(ppo_interaction_data, "ppo_interaction_effects_top20.csv", top_k=20)  # <--- save top 20 separately
        

# === MAIN EFFECTS ===
sac_main_data = aggregate_effects(sac_seed_dirs, main_file, parse_main_effects)
plot_error_bars(sac_main_data, " (b) FANOVA Main Effects for SAC (Mean ± Std across seeds)", "sac_main_effects_avg.png")
save_averaged_effects(sac_main_data, "sac_main_effects_avg.csv")  # <--- save to CSV

# === INTERACTION EFFECTS ===
sac_interaction_data = aggregate_effects(sac_seed_dirs, interaction_file, parse_interaction_effects)
plot_error_bars(sac_interaction_data, "(b) FANOVA Interaction Effects for SAC (Mean ± Std across seeds)", "sac_interaction_effects_avg.png")
plot_error_bars(sac_interaction_data, "(b) Top 20 FANOVA Interaction Effects for SAC (Mean ± Std across seeds)", "sac_interaction_effects_top20.png", top_k=20)
save_averaged_effects(sac_interaction_data, "sac_interaction_effects_avg.csv")  # <--- save to CSV
save_averaged_effects(sac_interaction_data, "sac_interaction_effects_top20.csv", top_k=20)  # <--- save top 20 separately


# === NORMALIZATION ===
# Combine both main and interaction importances for total sum
'''
# Compute mean effect per parameter
main_means = {k: np.mean(v) for k, v in main_data.items()}
interaction_means = {k: np.mean(v) for k, v in interaction_data.items()}

# Compute total sums
main_total = sum(main_means.values())
interaction_total = sum(interaction_means.values())
total = main_total + interaction_total

# Normalize
main_share = main_total / total
interaction_share = interaction_total / total

# Report

print(f"Main effects total : {main_total:.3f}")
print(f"Main effect share: {main_share:.3f}")

print(f"Interaction effect total : {interaction_total:.3f}")
print(f"Interaction effect share: {interaction_share:.3f}")
'''

# === NORMALIZATION WITH STD ===
main_shares = []
interaction_shares = []

for seed_idx in range(len(seed_dirs)):
    main_effects = []
    interaction_effects = []

    # Collect main effects for this seed
    for key, vals in main_data.items():
        if seed_idx < len(vals):
            main_effects.append(vals[seed_idx])

    # Collect interaction effects for this seed
    for key, vals in interaction_data.items():
        if seed_idx < len(vals):
            interaction_effects.append(vals[seed_idx])

    main_total = sum(main_effects)
    interaction_total = sum(interaction_effects)
    total = main_total + interaction_total

    main_shares.append(main_total / total)
    interaction_shares.append(interaction_total / total)

# Convert to arrays for easier stats
main_shares = np.array(main_shares)
interaction_shares = np.array(interaction_shares)

# Report with higher precision
print("Main shares per seed:", [f"{v:.6f}" for v in main_shares])
print("Interaction shares per seed:", [f"{v:.6f}" for v in interaction_shares])
print(f"Main share: {main_shares.mean():.6f} ± {main_shares.std():.6f}")
print(f"Interaction share: {interaction_shares.mean():.6f} ± {interaction_shares.std():.6f}")



# Perform paired t-test
t_stat, p_value = ttest_rel(interaction_shares, main_shares)

print(f"T-statistic: {t_stat:.4f}, p-value: {p_value:.6f}")




