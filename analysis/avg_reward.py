import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load your CSV
df1 = pd.read_csv("sac_sparse_cont.csv")
#trial.value=cost =-1*best_mean_reward
# Convert cost → reward 
df1["reward"] = -df1["value"]


# Box plot of reward  per seed
ax = df1.boxplot(
    column="reward",
    by="seed",
    grid=False,
    widths=0.40,        # thinner boxes (try 0.3–0.4)
    patch_artist=False   # keeps style consistent if you later color boxes
)


ax.set_title("(b) SAC", fontsize=15, pad=10)
ax.set_xlabel("Seed", fontsize=13, labelpad=6)
ax.set_ylabel("Best mean evaluation reward", fontsize=13, labelpad=6)

ax.tick_params(axis="both", labelsize=13)
plt.suptitle("")
plt.tight_layout()
plt.savefig("SAC_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()





# Group by seed and calculate boxplot stats
summary_stats = df1.groupby("seed")["reward"].describe(percentiles=[0.25, 0.5, 0.75])

# Extract only the relevant boxplot statistics
boxplot_stats = summary_stats[["min", "25%", "50%", "75%", "max"]]

# Print results
for seed, stats in boxplot_stats.iterrows():
    print(f"Seed {seed}:")
    print(f"  Min   : {stats['min']:.3f}")
    print(f"  Q1    : {stats['25%']:.3f}")
    print(f"  Median: {stats['50%']:.3f}")
    print(f"  Q3    : {stats['75%']:.3f}")
    print(f"  Max   : {stats['max']:.3f}")
    print("-" * 30)



# Load your CSV
df2 = pd.read_csv("ppo_sparse_cont.csv")
#trial.value=cost =-1*best_mean_reward
# Convert cost → reward 
df2["reward"] = -df2["value"]


# Box plot of reward  per seed
plt.figure(figsize=(8,6))

ax = df2.boxplot(
    column="reward",
    by="seed",
    grid=False,
    widths=0.40,        # thinner boxes (try 0.3–0.4)
    patch_artist=False   # keeps style consistent if you later color boxes
)



# Titles & labels — bigger fonts
ax.set_title("(a) PPO", fontsize=15, pad=10)
ax.set_xlabel("Seed", fontsize=13, labelpad=6)
ax.set_ylabel("Best mean evaluation reward", fontsize=13, labelpad=6)

# Tick label sizes
ax.tick_params(axis="both", labelsize=13)

# Remove pandas' default suptitle ("reward by seed")
plt.suptitle("")

plt.tight_layout()
plt.savefig("PPO_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()








# Group by seed and calculate boxplot stats
summary_stats = df2.groupby("seed")["reward"].describe(percentiles=[0.25, 0.5, 0.75])

# Extract only the relevant boxplot statistics
boxplot_stats = summary_stats[["min", "25%", "50%", "75%", "max"]]

# Print results
for seed, stats in boxplot_stats.iterrows():
    print(f"Seed {seed}:")
    print(f"  Min   : {stats['min']:.3f}")
    print(f"  Q1    : {stats['25%']:.3f}")
    print(f"  Median: {stats['50%']:.3f}")
    print(f"  Q3    : {stats['75%']:.3f}")
    print(f"  Max   : {stats['max']:.3f}")
    print("-" * 30)





