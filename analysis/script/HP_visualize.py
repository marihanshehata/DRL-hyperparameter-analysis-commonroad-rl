import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator




# ============================================================
# Visualization helpers
# ============================================================
def plot_numeric_hp(numeric_df, out_folder, algo,ordered_clusters):
    os.makedirs(out_folder, exist_ok=True)

    hp_list = numeric_df["hp"].unique()
    clusters = ordered_clusters
    #clusters = sorted(numeric_df["cluster"].unique())

    for hp in hp_list:
        fig, ax = plt.subplots(figsize=(16, 6))
        df_hp = (
            numeric_df[numeric_df["hp"] == hp]
            .set_index("cluster")
            .reindex(clusters)
        )

        #df_hp = numeric_df[numeric_df["hp"] == hp].set_index("cluster").loc[clusters]

        stats_list = []
        positions = list(range(len(clusters)))

        for _, row in df_hp.iterrows():
            stats_list.append({
                "med": float(row["median"]),
                "q1": float(row["q1"]),
                "q3": float(row["q3"]),
                "whislo": float(row["min"]),
                "whishi": float(row["max"]),
                "fliers": []
            })

        ax.bxp(
            stats_list,
            positions=positions,
            showfliers=False,
            widths=0.6
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(clusters, rotation=90, fontsize=10)

        ax.set_title(f"{algo.upper()} — Numeric HP: {hp}", fontsize=14, pad=20)
        ax.set_ylabel(hp, fontsize=14)
        ax.set_xlabel("Cluster ID", fontsize=12)

        save_path = os.path.join(out_folder, f"{hp}_boxplot.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"[Saved] {save_path}")


def plot_categorical_hp(categorical_df, out_folder, algo,ordered_clusters):
    os.makedirs(out_folder, exist_ok=True)

    hp_list = categorical_df["hp"].unique()
    clusters = ordered_clusters
    #clusters = sorted(categorical_df["cluster"].unique())

    for hp in hp_list:
        df_hp = categorical_df[categorical_df["hp"] == hp].copy()

        pct_cols = [c for c in df_hp.columns if c.startswith("pct_")]

        plot_records = []
        for _, row in df_hp.iterrows():
            cl = row["cluster"]
            for col in pct_cols:
                pct = row[col]
                if pct > 0:
                    plot_records.append({
                        "cluster": cl,
                        "category": col.replace("pct_", ""),
                        "pct": pct
                    })

        plot_df = pd.DataFrame(plot_records)

        fig, ax = plt.subplots(figsize=(16, 6))
        sns.barplot(
            data=plot_df,
            x="cluster",
            y="pct",
            hue="category",
            order=clusters,
            ax=ax
        )

        #sns.barplot(data=plot_df,x="cluster",y="pct",hue="category",ax=ax)

        ax.set_title(f"{algo.upper()} — Categorical HP: {hp}", fontsize=14, pad=20)
        ax.set_ylabel("Percentage (%)", fontsize=14)
        ax.set_xlabel("Cluster ID", fontsize=12)

        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels(clusters, rotation=90, fontsize=10)

        ax.legend(
            title="Category",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0
        ) 

        save_path = os.path.join(out_folder, f"{hp}_barchart.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"[Saved categorical] {save_path}")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac"])
    parser.add_argument("--base", type=str, default="paper2")
    args = parser.parse_args()

    algo = args.algo
    base = args.base

    input_folder = os.path.join(base, algo, "hp_summaries")
    out_folder = os.path.join(base, algo, "hp_plots")

    os.makedirs(out_folder, exist_ok=True)

    reward_file = os.path.join(base, algo, f"{algo}_with_clusters.csv")
    reward_df = pd.read_csv(reward_file)

    # ---------------------------------------------------
    # Compute reward-based cluster ordering (HIGH → LOW)
    # ---------------------------------------------------
    cluster_medians = (
        reward_df.groupby("cluster")["reward"]
        .median()
        .sort_values(ascending=False)
    )

    ordered_clusters = cluster_medians.index.tolist()

    numeric_file = os.path.join(input_folder, f"{algo}_numeric_summary.csv")
    categorical_file = os.path.join(input_folder, f"{algo}_categorical_summary.csv")

    numeric_df = pd.read_csv(numeric_file)
    categorical_df = pd.read_csv(categorical_file)

    print("\n=== Plotting Numeric HP ===")
    plot_numeric_hp(numeric_df, os.path.join(out_folder, "numeric"), algo,ordered_clusters)

    print("\n=== Plotting Categorical HP ===")
    plot_categorical_hp(categorical_df, os.path.join(out_folder, "categorical"), algo,ordered_clusters)
    print("Cluster order (high → low reward):", ordered_clusters)

    print("\nAll plots saved to:", out_folder)
if __name__ == "__main__":
    main()
