import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Load both CSV files
sac_df = pd.read_csv("sac_sparse_cont.csv")
ppo_df = pd.read_csv("ppo_sparse_cont.csv")

# Add an algorithm column to each
sac_df["algo"] = "SAC"
ppo_df["algo"] = "PPO"

# Concatenate
df = pd.concat([sac_df, ppo_df], ignore_index=True)

# Make the box plot
plt.figure(figsize=(6,6))
df.boxplot(column="value", by="algo", grid=False)

plt.title("Reward Distribution: SAC vs PPO")
plt.suptitle("")  # remove default subtitle
plt.xlabel("Algorithm")
plt.ylabel("Reward")
plt.show()





# Load trial results for each setting
files = {
    "SAC ": "sac_sparse_cont.csv",
    "PPO": "ppo_sparse_cont.csv"
    
}

# Dictionary to store summary statistics
summary = []

# Compute statistics for each setting
for setting, filepath in files.items():
    df = pd.read_csv(filepath)
    
    # Group by trial number to get average reward across seeds
    trial_means = df.groupby('trial_number')['value'].mean() * -1

    
    stats = {
        "Setting": setting,
        "Min Reward": trial_means.min(),
        "Max Reward": trial_means.max(),
        "Mean Reward": trial_means.mean(),
        "Std Dev": trial_means.std()
    }
    summary.append(stats)

# Create summary DataFrame
summary_df = pd.DataFrame(summary)

# Print and save to file
print(summary_df.to_string(index=False))
summary_df.to_csv("performance_statistics.csv", index=False)



# Combine all into one DataFrame
all_data = []

for setting, filepath in files.items():
    df = pd.read_csv(filepath)
    grouped = df.groupby('trial_number')['value'].mean().reset_index()
    grouped['value'] *= -1  # Correct the sign
    grouped['Setting'] = setting
    grouped.rename(columns={"value": "mean_reward"}, inplace=True)

    all_data.append(grouped)

plot_df = pd.concat(all_data, ignore_index=True)

# Set style
sns.set(style="whitegrid", font_scale=1.2)

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x="Setting", y="mean_reward", data=plot_df)

plt.title("Distribution of Mean Final Reward per Trial")
plt.ylabel("Mean Reward (over seeds)")

plt.tight_layout()
plt.savefig("reward_boxplot.png", dpi=300)
plt.show()



