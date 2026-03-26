import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def load_optuna_studies_to_df(
    log_path,
    databases: dict,
    user: str = "optuna",
    password: str = "pass",
    host: str = "localhost",
    port: int = 5432,
    study_name_template: str = "study_8env_seed{seed}",
    out_csv: str = "ppo_studies.csv",
    meta_cols: list = None,
) -> pd.DataFrame:
    """
    Load Optuna trials from multiple Postgres databases into a single DataFrame.

    Parameters
    ----------
    databases : dict
        Mapping from database name (str) to seed/int. Example:
            {"cs_seed0db": 0, "cs_seed42db": 42, ...}
    user, password, host, port : connection params for Postgres
    study_name_template : str
        Python format string for study name, must contain "{seed}".
        Example: "study_8env_seed{seed}".
    out_csv : str
        Path to save resulting CSV.
    meta_cols : list or None
        List of meta columns to treat specially (default ["seed","trial_number","params","value"]).

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per Optuna trial (flattened params).
    """
    if meta_cols is None:
        meta_cols = ["seed", "trial_number", "params", "value"]

    all_trials = []

    for db_name, seed in databases.items():
        storage_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        study_name = study_name_template.format(seed=seed)
        print(f" Loading study '{study_name}' from DB '{db_name}' ...")

        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
        except Exception as e:
            print(f"❌ Failed to load study '{study_name}' from '{db_name}': {e}")
            continue

        for trial in study.trials:
            # trial.value may be None (no result) — convert to NaN
            row = {
                "seed": seed,
                "value": trial.value if trial.value is not None else np.nan,
                "params": str(trial.params),
                "trial_number": trial.number,
            }
            # add param keys as columns (flatten)
            row.update(trial.params)
            all_trials.append(row)

    # Build DataFrame
    df = pd.DataFrame(all_trials)

    # Replace explicit None with NaN
    df = df.replace({None: np.nan})

    # Try to convert non-meta columns to numeric where possible (preserve strings)
    for col in df.columns:
        if col not in meta_cols:
            # keep behavior similar to your original code: try numeric conversion but do not force-convert strings away
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                # fallback: leave column as-is
                pass

    # Save and return
    df.to_csv(f"{log_path}/{out_csv}", index=False)
    print(f"✅ Saved {out_csv} with shape: {df.shape}")
    print("Columns:", list(df.columns))

    return df



def normalize_ppo_dataset(log_path):
    """
    Load PPO study CSV, detect numeric hyperparameters, normalize them,
    and save the normalized CSV.
    Returns the normalized dataframe.
    """

    # Load dataset
    df = pd.read_csv(log_path)
    print(" Loaded PPO dataset:", df.shape)

    # Meta columns that should NOT be normalized
    meta_cols = ["seed", "trial_number", "params", "value"]

    # Step 1: Identify hyperparameter columns
    hyper_cols = [c for c in df.columns if c not in meta_cols]

    # Step 2: Detect numeric hyperparameter columns
    numeric_hyper_cols = []
    for col in hyper_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_hyper_cols.append(col)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].notna().any():
                numeric_hyper_cols.append(col)

    print(" Numeric hyperparameters:", numeric_hyper_cols)

    # Step 3: Normalize numeric columns
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[numeric_hyper_cols] = scaler.fit_transform(df[numeric_hyper_cols])

    print("🔎 Normalized ranges:")
    print(df_norm[numeric_hyper_cols].describe().loc[["min", "max"]])

    # Step 4: Save normalized dataframe
    output_path = f"{log_path}/ppo_studies_normalized.csv"
    df_norm.to_csv(output_path, index=False)
    print(f"💾 Saved normalized PPO dataset → {output_path}")

    return df_norm




def cluster_ppo_trials(
    input_path="paper2/ppo/ppo_studies_normalized.csv",
    n_clusters=20,
    random_state=0
):
    """
    Load normalized PPO trial data, compute reward, detect numeric hyperparameters,
    run KMeans clustering, and compute cluster summary statistics.

    Returns:
        df (pd.DataFrame): full dataframe with cluster labels
        summary (pd.DataFrame): aggregated cluster statistics
        kmeans (KMeans): trained clustering model
    """

    print(" Loading normalized PPO dataset...")
    df = pd.read_csv(input_path)
    print(" Loaded PPO normalized dataframe:", df.shape)

    # Convert the Optuna objective (minimization) into reward
    df["reward"] = -df["value"]

    # Meta columns to exclude from clustering
    meta_cols = ["seed", "trial_number", "params", "value", "reward"]

    hyper_cols = [c for c in df.columns if c not in meta_cols]

    # Detect numeric parameter columns (PPO has only numeric)
    numeric_hyper_cols = []
    for col in hyper_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_hyper_cols.append(col)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].notna().any():
                numeric_hyper_cols.append(col)

    print(" Numeric hyperparameters:", numeric_hyper_cols)
    print(" Hyperparameters used for clustering:", hyper_cols)

    # Prepare feature matrix
    X = df[hyper_cols].values
    rewards = df["reward"].values

    # Run KMeans clustering
    print(f"🔍 Running KMeans with n_clusters={n_clusters} ...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    df["cluster"] = kmeans.fit_predict(X)

    # Aggregate cluster performance statistics
    summary = (
        df.groupby("cluster")
        .agg(
            median_reward=("reward", lambda x: np.nanmedian(x)),
            iqr_reward=("reward", lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
            q1_reward=("reward", lambda x: np.nanpercentile(x, 25)),
            q3_reward=("reward", lambda x: np.nanpercentile(x, 75)),
            count=("reward", "count")
        )
    )

    print("📊 Cluster Summary:")
    print(summary)

    return df, summary, kmeans



def plot_cluster_rewards(df, title="Reward Distribution per Cluster (PPO)",save_path=None):
    """
    Visualize reward distribution for each PPO cluster using a boxplot.

    Args:
        df (pd.DataFrame): DataFrame containing at least
            - 'cluster' column
            - 'reward' column
        title (str): Title for the plot
    """

    if "cluster" not in df.columns or "reward" not in df.columns:
        raise ValueError("DataFrame must contain 'cluster' and 'reward' columns.")

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="cluster", y="reward")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()    # prevents double display / memory leak
    else:
        plt.show()

def plot_ppo_cluster_rewards(
    df,
    title="Reward Distribution per Cluster (PPO)",
    save_path=None,
    cmap_name="coolwarm"
):
    """
    Visualize reward distribution for each PPO cluster using a boxplot,
    ordered from highest to lowest median reward and colored cold → warm.
    No legend / colorbar.
    """

    if "cluster" not in df.columns or "reward" not in df.columns:
        raise ValueError("DataFrame must contain 'cluster' and 'reward' columns.")

    # ---------------------------------------------------
    # 1. Compute median reward per cluster (DESCENDING)
    # ---------------------------------------------------
    cluster_medians = (
        df.groupby("cluster")["reward"]
        .median()
        .sort_values(ascending=False)
    )

    ordered_clusters = cluster_medians.index.tolist()

    # ---------------------------------------------------
    # 2. Normalize medians for colormap
    # ---------------------------------------------------
    norm = mcolors.Normalize(
        vmin=cluster_medians.min(),
        vmax=cluster_medians.max()
    )
    cmap = cm.get_cmap(cmap_name)

    palette = {
        cluster: cmap(norm(cluster_medians.loc[cluster]))
        for cluster in ordered_clusters
    }

    # ---------------------------------------------------
    # 3. Plot
    # ---------------------------------------------------
    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x="cluster",
        y="reward",
        order=ordered_clusters,
        palette=palette
    )

    plt.title(title)
    plt.xlabel("Cluster (Highest → Lowest Reward)")
    plt.ylabel("Reward")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # ---------------------------------------------------
    # 4. Save or show
    # ---------------------------------------------------
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
      


def compute_and_rank_ppo_clusters(df, summary, min_cluster_size=5):
    """
    Compute robustness scores for PPO clusters and rank them.

    Args:
        df (pd.DataFrame): The full PPO dataframe containing the 'cluster' and reward stats.
        summary (pd.DataFrame): Cluster summary with columns:
            ['median_norm', 'q1_norm', 'iqr_norm', 'count']
        min_cluster_size (int): Minimum cluster size required for eligibility.

    Returns:
        summary (pd.DataFrame): Updated summary with robustness score.
        top_clusters (pd.DataFrame): Top 10 ranked clusters.
        best_cluster (int): Index of the best cluster.
        best_cluster_data (pd.DataFrame): Subset of df for the best cluster.
    """

    # === Compute robustness score ===
    summary["robust_score"] = (
        0.45 * summary["median_reward"] +     # central performance
        0.45 * summary["q1_reward"] -         # pessimistic robustness
        0.10 * summary["iqr_reward"]          # variance penalty
    )

    # === Enforce minimum cluster size ===
    summary.loc[summary["count"] < min_cluster_size, "robust_score"] = -np.inf

    # === Rank clusters ===
    top_clusters = summary.sort_values("robust_score", ascending=False).head(20)
    print("\n Top clusters by robustness score:")
    #top_clusters = summary.sort_values("median_reward", ascending=False).head(20)
    #print("\n Top clusters by median reward:")
    
    print(top_clusters)
    
    # === Determine eligible clusters (that pass min size) ===
    eligible = summary[summary["robust_score"] > -np.inf]

    if eligible.empty:
        # No cluster satisfies minimum size; return with Nones
        return (
            summary,
            top_clusters,
            None, pd.DataFrame(),
            None, pd.DataFrame(),
            None, pd.DataFrame(),
        )

    # Sort eligible clusters by robustness: worst -> best
    eligible_sorted = eligible.sort_values("robust_score", ascending=True)



    # ---------- 2) Best-performing 2 clusters (best robustness) ----------
    best_cluster1 = eligible_sorted.index[-1]
    best_cluster1_data = df[df["cluster"] == best_cluster1]
    
    best_cluster2 = eligible_sorted.index[-2]
    best_cluster2_data = df[df["cluster"] == best_cluster2]

    # ---------- 3) Stable moderate cluster ----------
    # Just take the middle cluster in the sorted list.
    if len(eligible_sorted) >= 3:
        middle_pos = len(eligible_sorted) // 2
        stable_cluster = eligible_sorted.index[middle_pos]
    else:
        # If only 1 or 2 eligible clusters, just reuse best as "stable".
        stable_cluster = best_cluster

    stable_cluster_data = df[df["cluster"] == stable_cluster]

    return (
        summary,
        top_clusters,
        best_cluster1, best_cluster1_data,
        best_cluster2, best_cluster2_data,
        stable_cluster, stable_cluster_data
    )







def compute_ppo_configs_from_clusters(
    log_path,
    best_cluster1, best_cluster1_data,
    best_cluster2, best_cluster2_data,
    stable_cluster, stable_cluster_data,
    raw_csv_name="ppo_studies.csv",
    n_env=8,
):
    """
    For each given cluster (best1, best2, stable, poor):
      1. Take median normalized HP in that cluster
      2. Denormalize them to real values using MinMaxScaler fitted on raw data
      3. Auto-correct to PPO param ranges / dtypes
      4. Add nminibatches
    Returns:
        dict with configs for each cluster:
        {
            "best1": {...},
            "best2": {...},
            "stable": {...},
            "poor": {...},
        }
    """

    # ----------------------------------------------------
    # 1) Load raw HP data
    # ----------------------------------------------------
    raw_csv_path = os.path.join(log_path, raw_csv_name)
    df = pd.read_csv(raw_csv_path)
    #print("📄 Loaded PPO dataset:", df.shape)

    # ----------------------------------------------------
    # 2) Identify hyperparameter columns
    # ----------------------------------------------------
    meta_cols = ["seed", "trial_number", "params", "value"]
    hyper_cols = [c for c in df.columns if c not in meta_cols]

    # ----------------------------------------------------
    # 3) Detect numeric hyperparameter columns
    # ----------------------------------------------------
    numeric_hyper_cols = []
    for col in hyper_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_hyper_cols.append(col)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].notna().any():
                numeric_hyper_cols.append(col)

    #print("✅ Numeric hyperparameters:", numeric_hyper_cols)

    # Raw dataset for fitting MinMax scaler
    df_raw = df.copy()

    # Fit scaler on raw numeric hyperparameters
    scaler = MinMaxScaler()
    scaler.fit(df_raw[numeric_hyper_cols])

    # --------------------------------------------
    # Helper: from cluster data -> corrected config
    # --------------------------------------------
    param_ranges = {
        "batch_size": [8, 16, 32, 64, 128, 256, 512],
        "n_steps": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        "gamma": [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        "lr": (1e-5, 1.0),
        "ent_coef": (1e-8, 0.1),
        "cliprange": [0.1, 0.2, 0.3, 0.4],
        "noptepochs": [1, 5, 10, 20, 30, 50],
        "lamdba": [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0],
        "max_grad_norm": (0.0, 1.0),
        "vf_coef": (0.0, 1.0),
        "cliprange_vf": (0.0, 0.5),
    }

    def _cluster_to_config(cluster_data):
        # 1) Median hyperparameters in this cluster (still normalized)
        robust_med = cluster_data[hyper_cols].median().to_dict()

        # 2) Denormalize numeric hyperparameters
        robust_med_df = pd.DataFrame([robust_med])[numeric_hyper_cols]
        real_values = scaler.inverse_transform(robust_med_df)
        real_values = pd.DataFrame(real_values, columns=numeric_hyper_cols)

        final_config = real_values.iloc[0].to_dict()

        # 3) Auto-correct to PPO ranges
        corrected = final_config.copy()
        for k, v in final_config.items():
            if k not in param_ranges:
                continue

            allowed = param_ranges[k]

            # Discrete / categorical
            if isinstance(allowed, list):
                nearest = min(allowed, key=lambda x: abs(x - v))
                corrected[k] = nearest

            # Continuous range
            elif isinstance(allowed, tuple):
                lo, hi = allowed
                corrected[k] = max(lo, min(v, hi))

        final_config = corrected

        # 4) Compute nminibatches if possible
        if "n_steps" in final_config and "batch_size" in final_config:
            n_steps = int(final_config["n_steps"])
            batch_size = int(final_config["batch_size"])
            rollout_size = n_steps * n_env
            if batch_size > 0:
                n_minibatches = max(1, rollout_size // batch_size)
            else:
                n_minibatches = 1
            final_config["nminibatches"] = n_minibatches

        return final_config

    def rename_ppo_keys(config_dict):
        """
        Rename PPO hyperparameter keys consistently across all configs.
        """
        rename_map = {
            "lr": "learning_rate",
            "lamdba": "lam",
            "batch_size":"#batch_size"
        }

        new_dict = {}
        for k, v in config_dict.items():
            new_key = rename_map.get(k, k)   # rename if in map, else keep original
            new_dict[new_key] = v

        return new_dict

    # --------------------------------------------
    # Build configs for each cluster
    # --------------------------------------------
    best1_config = rename_ppo_keys(_cluster_to_config(best_cluster1_data))
    best2_config = rename_ppo_keys(_cluster_to_config(best_cluster2_data))
    stable_config = rename_ppo_keys(_cluster_to_config(stable_cluster_data))


    # Optional: nice printing
    print("\n🚀 Final PPO configs (real-scale values):")

    print(f"\nBest cluster 1 (id={best_cluster1}):")
    for k, v in best1_config.items():
        print(f"  {k}: {v}")

    print(f"\nBest cluster 2 (id={best_cluster2}):")
    for k, v in best2_config.items():
        print(f"  {k}: {v}")

    print(f"\nStable cluster (id={stable_cluster}):")
    for k, v in stable_config.items():
        print(f"  {k}: {v}")


    # Sanity check range print
    print("\n📊 Range check (original data min–max):")
    print(df_raw[numeric_hyper_cols].agg(["min", "max"]).T)

    return {
        "best1": best1_config,
        "best2": best2_config,
        "stable": stable_config
    }



def denormalize_entire_cluster(
    log_path,
    raw_csv_name="ppo_studies.csv",
    cluster_df=None
):
    """
    Denormalizes ALL hyperparameter rows of a given PPO cluster_df.
    - Uses MinMaxScaler fitted on raw real-scale PPO CSV.
    - Applies PPO valid ranges.
    - Does NOT rename any keys.
    - Works exactly like the SAC version you approved.
    """

    #print("\n================= PPO denormalize_entire_cluster() =================")
    #print(f"log_path: {log_path}")
    #print(f"raw_csv_name: {raw_csv_name}")

    # -----------------------------
    # 1. Validate cluster_df
    # -----------------------------
    if cluster_df is None:
        print("❌ ERROR: cluster_df is None.")
        return None

    #print(f"cluster_df rows: {len(cluster_df)}")
    if len(cluster_df) == 0:
        print("❌ ERROR: cluster_df is EMPTY.")
        return None

    #print(f"cluster_df columns: {list(cluster_df.columns)}")

    # -----------------------------
    # 2. Load raw PPO CSV
    # -----------------------------
    raw_path = f"{log_path}/{raw_csv_name}"
    #print(f"Loading raw file: {raw_path}")

    try:
        df_raw = pd.read_csv(raw_path)
        #print(f"✔ Loaded PPO raw CSV: {df_raw.shape}")
    except Exception as e:
        print(f"❌ ERROR loading raw CSV: {e}")
        return None

    # -----------------------------
    # 3. Identify column groups
    # -----------------------------
    meta_cols = ["seed", "trial_number", "params", "value"]
    categorical_cols = []   # PPO has none except maybe cliprange, but treat as numeric

    numeric_cols = [
        c for c in df_raw.columns
        if c not in meta_cols + categorical_cols
    ]

    #print(f"Numeric columns detected: {numeric_cols}")

    # Numeric conversion
    for col in numeric_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # -----------------------------
    # 4. Fit scaler
    # -----------------------------
    #print("Fitting MinMaxScaler ...")
    try:
        scaler = MinMaxScaler()
        scaler.fit(df_raw[numeric_cols])
        #print("✔ Scaler fit OK.")
    except Exception as e:
        print(f"❌ ERROR fitting scaler: {e}")
        return None

    # -----------------------------
    # 5. PPO valid param ranges
    # -----------------------------
    #print("Loading PPO parameter ranges...")

    param_ranges = {
        "batch_size": [8, 16, 32, 64, 128, 256, 512],
        "n_steps": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        "gamma": [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        "lr": (1e-5, 1.0),
        "ent_coef": (1e-8, 0.1),
        "cliprange": [0.1, 0.2, 0.3, 0.4],
        "noptepochs": [1, 5, 10, 20, 30, 50],
        "lamdba": [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0],
        "max_grad_norm": (0.0, 1.0),
        "vf_coef": (0.0, 1.0),
        "cliprange_vf": (0.0, 0.5),
    }

    # -----------------------------
    # 6. Denormalize numeric columns
    # -----------------------------
    #print("Extracting normalized numeric values from cluster_df...")

    try:
        norm_numeric = cluster_df[numeric_cols]
        #print(f"norm_numeric shape: {norm_numeric.shape}")
    except Exception as e:
        print(f"❌ ERROR extracting numeric cluster columns: {e}")
        return None

    #print("Applying scaler.inverse_transform() ...")

    try:
        denorm_array = scaler.inverse_transform(norm_numeric)
        #print("✔ Denormalization OK.")
    except Exception as e:
        print(f"❌ ERROR in inverse_transform: {e}")
        return None

    denorm_numeric_df = pd.DataFrame(denorm_array, columns=numeric_cols)
    #print(f"Denormalized numeric df shape: {denorm_numeric_df.shape}")

    # -----------------------------
    # 7. Add categorical columns
    # -----------------------------
    final_df = denorm_numeric_df.copy()
    for col in categorical_cols:
        final_df[col] = cluster_df[col] if col in cluster_df else None

    # -----------------------------
    # 8. Snap values to legal PPO ranges
    # -----------------------------
    #print("Snapping values to PPO ranges...")

    def snap_to_range(key, val):
        allowed = param_ranges.get(key, None)
        if allowed is None:
            return val

        if isinstance(allowed, list):
            return min(allowed, key=lambda x: abs(x - val))

        lo, hi = allowed
        return max(lo, min(val, hi))

    for col in final_df.columns:
        if col in param_ranges:
            final_df[col] = final_df[col].apply(lambda v: snap_to_range(col, v))

    # -----------------------------
    # 9. Add metadata back
    # -----------------------------
    for col in meta_cols:
        if col in cluster_df.columns:
            final_df[col] = cluster_df[col].values

    if "cluster" in cluster_df.columns:
        final_df["cluster"] = cluster_df["cluster"].values

    #print(f"✔ FINAL DF SHAPE: {final_df.shape}")
    #print("================= END PPO denormalization =================\n")

    return final_df



def cluster_data(df, summary, target_cluster):


    # === Return cluster data for the requested ID ===
    if target_cluster not in summary.index:
        print(f"\n❌ Cluster {target_cluster} does not exist.")
        target_cluster_data = pd.DataFrame()
    else:
        target_cluster_data = df[df["cluster"] == target_cluster]
        print(f"\n📌 Extracted data for cluster {target_cluster}: {target_cluster_data.shape}")

    return  target_cluster_data

import os
import pandas as pd

def compute_configs_from_cluster(cluster_csv_path, n_env=8):
    """
    Input:
        cluster_csv_path : path to a SINGLE cluster CSV file (denormalized HP values)
    
    What it does:
        1. Loads cluster data
        2. Computes median for all hyperparameters
        3. Renames keys (lr → learning_rate, lamdba → lam, batch_size → #batch_size)
        4. Computes nminibatches
        5. Saves output as:
              ppo_<cluster_id>_configs.csv
    """

    # -------------------------------------------------------
    # 1. Load cluster CSV
    # -------------------------------------------------------
    cluster_df = pd.read_csv(cluster_csv_path)

    # Extract cluster number
    base = os.path.basename(cluster_csv_path)
    cluster_id = base.replace("cluster_", "").replace("_data.csv", "")

    # -------------------------------------------------------
    # 2. List of PPO hyperparameters expected
    # -------------------------------------------------------
    ppo_hps = [
        "batch_size", "n_steps", "gamma", "lr", "ent_coef", "cliprange",
        "noptepochs", "lamdba", "max_grad_norm", "vf_coef", "cliprange_vf"
    ]

    # Keep only available columns
    hp_present = [h for h in ppo_hps if h in cluster_df.columns]

    # -------------------------------------------------------
    # 3. Compute medians (real-scale)
    # -------------------------------------------------------
    medians = cluster_df[hp_present].median().to_dict()

    # -------------------------------------------------------
    # 4. Compute nminibatches
    # -------------------------------------------------------
    if "batch_size" in medians and "n_steps" in medians:
        n_steps = int(medians["n_steps"])
        batch_size = int(medians["batch_size"])
        rollout_size = n_steps * n_env
        nminibatches = max(1, rollout_size // batch_size)
    else:
        nminibatches = 1

    medians["nminibatches"] = nminibatches

    # -------------------------------------------------------
    # 5. Key renaming
    # -------------------------------------------------------
    def rename_ppo_keys(config_dict):
        rename_map = {
            "lr": "learning_rate",
            "lamdba": "lam",
            "batch_size": "#batch_size"
        }
        new_dict = {}
        for k, v in config_dict.items():
            new_key = rename_map.get(k, k)
            new_dict[new_key] = v
        return new_dict

    final_config = rename_ppo_keys(medians)

    # -------------------------------------------------------
    # 6. Save as ppo_<cluster>_configs.csv
    # -------------------------------------------------------
    out_filename = f"ppo_{cluster_id}_configs.csv"
    df_out = pd.DataFrame([final_config])

    # Ensure column order as you requested
    desired_order = [
        "#batch_size","n_steps","gamma","learning_rate","ent_coef",
        "cliprange","noptepochs","lam","max_grad_norm","vf_coef",
        "cliprange_vf","nminibatches"
    ]
    df_out = df_out.reindex(columns=desired_order)

    df_out.to_csv(out_filename, index=False)

    # Save in the same directory as the input file
    input_dir = os.path.dirname(cluster_csv_path)
    out_filename = os.path.join(input_dir, f"ppo_{cluster_id}_configs.csv")

    df_out.to_csv(out_filename, index=False)

    print(f" Saved PPO config to {out_filename}\n")
    print(df_out)
    print("\n Cluster configuration (YAML-style):")
    for k, v in final_config.items():
        print(f"  {k}: {v}")

    return final_config



def main():

    log_path = "paper2/ppo"
    os.makedirs(log_path, exist_ok=True)

    # connection params
    user = "optuna"
    password = "pass"
    host = "localhost"
    port = 5432

    ppo_databases = {
        "cs_seed0db": 0,
        "cs_seed42db": 42,
        "cs_seed123db": 123,
        "cs_seed999db": 999,
        "cs_seed2024db": 2024,
    }

    #load studies and convert to dadaframe

    '''
    ppo_df = load_optuna_studies_to_df(
        log_path,
        databases=ppo_databases,
        user=user,
        password=password,
        host=host,
        port=port,
        study_name_template="study_8env_seed{seed}",
        out_csv="ppo_studies.csv"
    )
    
    #normalize   
    df_norm = normalize_ppo_dataset(log_path)
    '''
    #clsuter 
    df_clusters, summary, model = cluster_ppo_trials(n_clusters=20)

    summary.to_csv(f"{log_path}/ppo_cluster_summary.csv")
    df_clusters.to_csv(f"{log_path}/ppo_with_clusters.csv")
    
    #plot clusters
    #plot_ppo_cluster_rewards(df_clusters, save_path=f"{log_path}/cluster_plot.png")
    plot_ppo_cluster_rewards(
        df_clusters,
        title="PPO Cluster Reward Distributions (Colored by Median Reward)",
        save_path=f"{log_path}/ppo_cluster_plot.png"
        )
    # rank clusters and choose (best-middle-worst)
    (
        summary_out,
        top_clusters,
        best_cluster1, best_cluster1_data,
        best_cluster2, best_cluster2_data,
        stable_cluster, stable_cluster_data,
    ) = compute_and_rank_ppo_clusters(df_clusters, summary, min_cluster_size=5)

    # Print results
    #print("\nBest cluster1:", best_cluster1,"Best cluster2:", best_cluster2)
    #print("Stable cluster:", stable_cluster)
    

    '''
    #denormalize and extract HP real scale values for each choosen cluster
    configs = compute_ppo_configs_from_clusters(
        log_path=log_path,
        best_cluster1=best_cluster1,
        best_cluster1_data=best_cluster1_data,
        best_cluster2=best_cluster2,
        best_cluster2_data=best_cluster2_data,
        stable_cluster=stable_cluster,
        stable_cluster_data=stable_cluster_data,
        raw_csv_name="ppo_studies.csv",   # or your actual file name
        n_env=8,
    )
    '''
    
    #-----save denormalized real scale data for each selstced cluster --------
    

        # Process clusters 0 → 19
    clusters_to_process = list(range(0, 20))

    for cid in clusters_to_process:
        print(f"\n==============================")
        print(f"🔍 Processing cluster {cid}")
        print(f"==============================")

        # ---------------------------------------------
        # 1. Extract cluster data (normalized)
        # ---------------------------------------------
        cluster_df = cluster_data(df_clusters, summary, cid)

        # ---------------------------------------------
        # 2. Denormalize entire cluster
        # ---------------------------------------------
        cluster_df = denormalize_entire_cluster(
            log_path="paper2/ppo",
            raw_csv_name="ppo_studies.csv",
            cluster_df=cluster_df
        )

        # ---------------------------------------------
        # 3. Save cluster CSV file
        # ---------------------------------------------
        out_csv = f"{log_path}/cluster_{cid}_data.csv"
        cluster_df.to_csv(out_csv, index=False)
        print(f" Saved denormalized cluster CSV: {out_csv}")

        # ---------------------------------------------
        # 4. Compute PPO configs for this cluster
        # ---------------------------------------------
        compute_configs_from_cluster(out_csv, n_env=8)

    

if __name__ == "__main__":
    main()
