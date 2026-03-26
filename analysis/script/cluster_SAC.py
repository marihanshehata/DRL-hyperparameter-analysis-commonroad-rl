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
    out_csv: str = "sac_studies.csv",
    meta_cols: list = None,
) -> pd.DataFrame:
    """
    Load Optuna trials from multiple Postgres databases into a single DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per Optuna trial (flattened params).
    """
    if meta_cols is None:
        meta_cols = ["seed", "trial_number", "params", "value"]
    
    exclude_cols = ["net_arch", "target_entropy","ent_coef"]  # exclude these always

    all_trials = []

    for db_name, seed in databases.items():
        storage_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        study_name = "study"
        print(f" Loading study '{study_name}' from DB '{db_name}' ...")

        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
        except Exception as e:
            print(f" Failed to load study '{study_name}' from '{db_name}': {e}")
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
        if col not in meta_cols and col not in exclude_cols:
            # keep behavior similar to your original code: try numeric conversion but do not force-convert strings away
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                # fallback: leave column as-is
                pass

    # Save and return
    df.to_csv(f"{log_path}/{out_csv}", index=False)
    print(f" Saved {out_csv} with shape: {df.shape}")
    print("Columns:", list(df.columns))

    return df



def normalize_sac_dataset(log_path):
    """
    Load SAC study CSV, detect numeric hyperparameters, normalize them,
    and save the normalized CSV.
    Returns the normalized dataframe.
    """

    # Load dataset
    df = pd.read_csv(log_path)
    print("📄 Loaded SAC dataset:", df.shape)

    # Meta columns that should NOT be normalized
    meta_cols = ["seed", "trial_number", "params", "value"]


    # Step 1: Identify hyperparameter columns
    hyper_cols = [c for c in df.columns if c not in meta_cols]
    exclude_cols = ["net_arch", "target_entropy","ent_coef"]  # exclude these always

    # Step 2: Detect numeric hyperparameter columns
    numeric_hyper_cols = []
    for col in hyper_cols:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_hyper_cols.append(col)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].notna().any():
                numeric_hyper_cols.append(col)

    print(" Numeric hyperparameters:", numeric_hyper_cols)
    print(" Excluded non-numeric:", set(hyper_cols) - set(numeric_hyper_cols))


    # Step 3: Normalize numeric columns
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[numeric_hyper_cols] = scaler.fit_transform(df[numeric_hyper_cols])

    print("🔎 Normalized ranges:")
    print(df_norm[numeric_hyper_cols].describe().loc[["min", "max"]])
    
    # Step 4: Save normalized dataframe
    output_path = f"{log_path}/sac_studies_normalized.csv"
    df_norm.to_csv(output_path, index=False)
    print(f" Saved normalized SAC dataset → {output_path}")

    return df_norm




def cluster_sac_trials(
    input_path="paper2/sac/sac_studies_normalized.csv",
    n_clusters=20,
    random_state=0
):
    """
    Load normalized sac trial data, compute reward, detect numeric hyperparameters,
    run KMeans clustering, and compute cluster summary statistics.

    Returns:
        df (pd.DataFrame): full dataframe with cluster labels
        summary (pd.DataFrame): aggregated cluster statistics
        kmeans (KMeans): trained clustering model
    """

    print(" Loading normalized sac dataset...")
    df = pd.read_csv(input_path)
    print(" Loaded sac normalized dataframe:", df.shape)

    # Convert the Optuna objective (minimization) into reward
    df["reward"] = -df["value"]

    # Meta columns to exclude from clustering
    meta_cols = ["seed", "trial_number", "params", "value", "reward"]
    exclude_cols = ["net_arch", "target_entropy", "ent_coef"]  # categorical/mixed
    
    # Keep only numeric hyperparameters
    hyper_cols = [c for c in df.columns if c not in meta_cols + exclude_cols]
    print(" Hyperparameters used for clustering:", hyper_cols)


    # Prepare feature matrix
    X = df[hyper_cols].values
    rewards = df["reward"].values

    # Run KMeans clustering
    print(f" Running KMeans with n_clusters={n_clusters} ...")
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

    print(" Cluster Summary:")
    print(summary)

    return df, summary, kmeans



def plot_cluster_rewards(df, title="Reward Distribution per Cluster (SAC)",save_path=None):


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


def plot_SAC_cluster_rewards(
    df,
    title="Reward Distribution per Cluster (SAC)",
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

def compute_and_rank_sac_clusters(df, summary, min_cluster_size=5):
    

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
        stable_cluster, stable_cluster_data,
    )

import ast

def extract_categorical_from_params(cluster_df):
    """
    Extract categorical SAC hyperparameters from the 'params' column
    and add them as proper dataframe columns.
    """

    categorical_cols = ["net_arch", "ent_coef", "target_entropy"]

    # Ensure params column exists
    if "params" not in cluster_df.columns:
        raise ValueError(" 'params' column not found in cluster_df")

    # Initialize columns
    for col in categorical_cols:
        cluster_df[col] = None

    # Extract values row by row
    for idx, p in cluster_df["params"].items():
        if pd.isna(p):
            continue

        try:
            d = ast.literal_eval(p)
        except Exception as e:
            print(f" Failed to parse params at row {idx}: {e}")
            continue

        for col in categorical_cols:
            if col in d:
                cluster_df.at[idx, col] = d[col]

    return cluster_df


def compute_configs_from_cluster(cluster_csv_path):
    """
    Load SAC cluster CSV (denormalized), compute:
        - numeric medians + range correction
        - categorical modes (robust version)
        - rename lr → learning_rate
        - save sac_<cluster_id>_configs.csv
        - print YAML-style summary
    """
    import os
    import pandas as pd

    # -------------------------------------------------------
    # Load data
    # -------------------------------------------------------
    cluster_df = pd.read_csv(cluster_csv_path)

    base = os.path.basename(cluster_csv_path)
    cluster_id = base.replace("cluster_", "").replace("_data.csv", "")

    # -------------------------------------------------------
    # SAC hyperparameter definitions
    # -------------------------------------------------------
    numeric_hps = [
        "gamma", "lr", "tau", "batch_size", "buffer_size",
        "learning_starts", "train_freq", "gradient_steps"
    ]

    categorical_hps = ["net_arch", "target_entropy", "ent_coef"]

    # Keep only those present in the CSV
    num_present = [h for h in numeric_hps if h in cluster_df.columns]
    cat_present = [h for h in categorical_hps if h in cluster_df.columns]

    # -------------------------------------------------------
    # SAC parameter ranges (with correction)
    # -------------------------------------------------------
    param_ranges = {
        "gamma":            [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        "lr":               (1e-5, 1.0),
        "tau":              (1e-4, 1.0),
        "batch_size":       [16, 32, 64, 128, 256, 512],
        "buffer_size":      [int(1e4), int(1e5), int(1e6)],
        "learning_starts":  (0, 10000),
        "train_freq":       (1, 1000),
        "gradient_steps":   [1, 10, 100, 300],
    }

    # -------------------------------------------------------
    # Compute medians for numeric
    # -------------------------------------------------------
    config = {}

    for hp in num_present:
        median_val = cluster_df[hp].median()

        allowed = param_ranges.get(hp)

        # Discrete
        if isinstance(allowed, list):
            try:
                corrected = min(allowed, key=lambda x: abs(x - median_val))
            except:
                corrected = allowed[0]
        # Continuous
        elif isinstance(allowed, tuple):
            lo, hi = allowed
            corrected = max(lo, min(hi, median_val))
        else:
            corrected = median_val

        config[hp] = corrected

    # -------------------------------------------------------
    # Robust categorical mode extraction (FIXED)
    # -------------------------------------------------------
    for hp in cat_present:
        if hp in cluster_df:
            counts = cluster_df[hp].value_counts()
            config[hp] = counts.idxmax() if not counts.empty else None
        else:
            config[hp] = None

    # -------------------------------------------------------
    # Rename SAC keys
    # -------------------------------------------------------
    def rename_sac_keys(d):
        rename_map = {"lr": "learning_rate"}
        return {rename_map.get(k, k): v for k, v in d.items()}

    final_config = rename_sac_keys(config)

    # -------------------------------------------------------
    # Save CSV next to input file
    # -------------------------------------------------------
    out_path = os.path.join(
        os.path.dirname(cluster_csv_path),
        f"sac_{cluster_id}_configs.csv"
    )

    pd.DataFrame([final_config]).to_csv(out_path, index=False)

    print(f" Saved SAC config to {out_path}\n")
    print(pd.DataFrame([final_config]))

    # -------------------------------------------------------
    # YAML-style pretty print
    # -------------------------------------------------------
    print("\n SAC Cluster Configuration (YAML-style):")
    for k, v in final_config.items():
        print(f"  {k}: {v}")

    return final_config



def compute_sac_configs_from_clusters(
    log_path,
    best_cluster1, best_cluster1_data,
    best_cluster2, best_cluster2_data,
    stable_cluster, stable_cluster_data,
    raw_csv_name="sac_studies.csv",
    n_env=1,
):
    """
    Computes SAC hyperparameter configurations for 4 clusters:
    best1, best2, stable, poor.

    Returns real-scale, range-corrected configurations.
    """

    # -----------------------------
    # 0. Definitions
    # -----------------------------
    meta_cols = ["seed", "trial_number", "params", "value","reward"]
    exclude_cols = ["net_arch", "target_entropy", "ent_coef"]

    # Load raw unnormalized CSV
    df_raw = pd.read_csv(f"{log_path}/{raw_csv_name}")

    # Identify numeric hyperparameters
    numeric_hyper_cols = [
        c for c in df_raw.columns if c not in meta_cols + exclude_cols
    ]

    # Ensure numeric conversion
    for col in numeric_hyper_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # Fit MinMaxScaler on original (real-scale) data
    scaler = MinMaxScaler()
    scaler.fit(df_raw[numeric_hyper_cols])

    # SAC parameter valid ranges
    param_ranges = {
        "gamma": [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        "lr": (1e-5, 1.0),
        "tau": (1e-4, 1.0),
        "batch_size": [16, 32, 64, 128, 256, 512],
        "buffer_size": [int(1e4), int(1e5), int(1e6)],
        "learning_starts": (0, 10000),
        "train_freq": (1, 1000),
        "gradient_steps": [1, 10, 100, 300]
    }

    # -----------------------------
    # Helper function for one cluster
    # -----------------------------
    def build_config(cluster_name, cluster_df):
        print(f"\n\n===== Processing cluster: {cluster_name} =====")

        # Step 1: Compute normalized medians (numeric HP only)
        robust_numeric = cluster_df[numeric_hyper_cols].median().to_dict()

        # Step 2: Determine categorical modes
        final_cat = {}
        for col in exclude_cols:
            if col in cluster_df.columns:
                counts = cluster_df[col].value_counts()
                final_cat[col] = counts.idxmax() if not counts.empty else None
            else:
                final_cat[col] = None

        # Step 3: Denormalize numeric values
        robust_df = pd.DataFrame([robust_numeric])[numeric_hyper_cols]
        denorm = scaler.inverse_transform(robust_df)
        denorm = pd.Series(denorm[0], index=numeric_hyper_cols).to_dict()

        # Step 4: Merge numeric + categorical
        real_cfg = denorm.copy()
        for col in exclude_cols:
            real_cfg[col] = final_cat[col]

        # Step 5: Correct values using SAC valid ranges
        corrected = real_cfg.copy()

        for k, v in real_cfg.items():
            if k not in param_ranges:
                continue

            allowed = param_ranges[k]

            # Discrete/categorical
            if isinstance(allowed, list):
                try:
                    corrected[k] = min(allowed, key=lambda x: abs(x - v))
                except:
                    corrected[k] = allowed[0]  # fallback

            # Continuous (tuple range)
            elif isinstance(allowed, tuple):
                lo, hi = allowed
                corrected[k] = max(lo, min(v, hi))

        
        
        
        
        
        return corrected
    
    
    

    # -----------------------------
    # Build configs for each cluster
    # -----------------------------
    best1_config = build_config(best_cluster1, best_cluster1_data)
    best2_config = build_config(best_cluster2, best_cluster2_data)
    stable_config = build_config(stable_cluster, stable_cluster_data)

    # -----------------------------
    # Return dict
    # -----------------------------
    return {
        "best1": best1_config,
        "best2": best2_config,
        "stable": stable_config,
    }


def denormalize_entire_cluster(
    log_path,
    raw_csv_name="sac_studies.csv",
    cluster_df=None
):
    """
    Denormalizes ALL hyperparameter rows of a given cluster_df.
    Unlike previous functions, this works on the full dataset,
    not only medians.
    """
    #print("\n================= denormalize_entire_cluster_debug() =================")
    #print(f"log_path: {log_path}")
    #print(f"raw_csv_name: {raw_csv_name}")



    # -----------------------------
    # 0. Definitions
    # -----------------------------
    meta_cols = ["seed", "trial_number", "params", "value", "reward"]
    categorical_cols = ["net_arch", "target_entropy", "ent_coef"]

    # Load raw unnormalized CSV
    df_raw = pd.read_csv(f"{log_path}/{raw_csv_name}")

    # Identify numeric HP columns
    numeric_cols = [c for c in df_raw.columns if c not in meta_cols + categorical_cols]
    #print(f"Numeric columns detected: {numeric_cols}")
    numeric_present = [c for c in numeric_cols if c in cluster_df.columns]


    # Ensure numeric types
    for col in numeric_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # Fit MinMaxScaler on real-scale data
    scaler = MinMaxScaler()
    scaler.fit(df_raw[numeric_cols])

    # -----------------------------
    # SAC allowed parameter ranges
    # -----------------------------
    param_ranges = {
        "gamma": [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        "lr": (1e-5, 1.0),
        "tau": (1e-4, 1.0),
        "batch_size": [16, 32, 64, 128, 256, 512],
        "buffer_size": [int(1e4), int(1e5), int(1e6)],
        "learning_starts": (0, 10000),
        "train_freq": (1, 1000),
        "gradient_steps": [1, 10, 100, 300]
    }

    # -----------------------------
    # 1. Denormalize numeric HPs
    # -----------------------------
    norm_numeric = cluster_df[numeric_present]
    #print(f"norm_numeric shape: {norm_numeric.shape}")

    denorm_array = scaler.inverse_transform(norm_numeric)
    denorm_numeric_df = pd.DataFrame(denorm_array,columns=numeric_present,index=cluster_df.index)

    #print(f"Denormalized numeric dataframe shape: {denorm_numeric_df.shape}")

    # -----------------------------
    # 2. Handle categorical variables
    # -----------------------------
    cluster_df = cluster_df.copy()
    

    
    for col in numeric_present:
        cluster_df[col] = denorm_numeric_df[col]

    

    # -----------------------------
    # 3. Snap values to SAC valid ranges
    # -----------------------------
    def snap_to_range(key, val):

        allowed = param_ranges.get(key, None)
        if allowed is None:
            return val

        # List means discrete candidates (gamma, batch_size, etc.)
        if isinstance(allowed, list):
            return min(allowed, key=lambda x: abs(x - val))

        # Tuple means continuous range
        lo, hi = allowed
        return max(lo, min(val, hi))

    for col in numeric_present:
        if col in param_ranges:
            cluster_df[col] = cluster_df[col].apply(lambda v: snap_to_range(col, v) )
    #print("Snapping complete.")

    '''
    # -----------------------------
    # 4. Add back metadata columns
    # -----------------------------
    for col in meta_cols:
        if col in cluster_df.columns:
            final_df[col] = cluster_df[col].values

    # Keep cluster assignment
    if "cluster" in cluster_df.columns:
        final_df["cluster"] = cluster_df["cluster"].values
    '''
    #print(f"✔ FINAL DF SHAPE: {final_df.shape}")
    #print("================= END DEBUG =================\n")
    return cluster_df







def rename_sac_keys(config_dict):
    """
    Rename sac hyperparameter keys consistently across all configs.
    """
    rename_map = {"lr": "learning_rate"}

    new_dict = {}
    for k, v in config_dict.items():
        new_key = rename_map.get(k, k)   # rename if in map, else keep original
        new_dict[new_key] = v

    return new_dict



def cluster_data(df, summary, target_cluster):


    # === Return cluster data for the requested ID ===
    if target_cluster not in summary.index:
        print(f"\n Cluster {target_cluster} does not exist.")
        target_cluster_data = pd.DataFrame()
    else:
        target_cluster_data = df[df["cluster"] == target_cluster]
        print(f"\n Extracted data for cluster {target_cluster}: {target_cluster_data.shape}")

    return  target_cluster_data





def main():

    log_path = "paper2/sac"
    os.makedirs(log_path, exist_ok=True)

    # connection params
    user = "optuna"
    password = "pass"
    host = "localhost"
    port = 5432

    sac_databases = {
        "sac_sparse_seed0": 0,
        "sac_sparse_seed42": 42,
        "sac_sparse_seed123": 123,
        "sac_sparse_seed999": 999,
        "sac_sparse_seed2024": 2024,
    }

    #load studies and convert to dadaframe

    '''
    sac_df = load_optuna_studies_to_df(
        log_path,
        databases=sac_databases,
        user=user,
        password=password,
        host=host,
        port=port,
        out_csv="sac_studies.csv"
    )
    '''
    
    #normalize   
    #df_norm = normalize_sac_dataset(log_path)
    
    #clsuter 
    df_clusters, summary, model = cluster_sac_trials(n_clusters=20)

    summary.to_csv(f"{log_path}/sac_cluster_summary.csv")
    df_clusters.to_csv(f"{log_path}/sac_with_clusters.csv")
    
    #plot clusters
    #plot_sac_cluster_rewards(df_clusters, save_path=f"{log_path}/cluster_plot.png")
    plot_SAC_cluster_rewards(
        df_clusters,
        title="SAC Cluster Reward Distributions (Colored by Median Reward)",
        save_path=f"{log_path}/SAC_cluster_plot.png"
    )
    # rank clusters and choose (best-middle-worst)
    (
        summary_out,
        top_clusters,
        best_cluster1, best_cluster1_data,
        best_cluster2, best_cluster2_data,
        stable_cluster, stable_cluster_data
    ) = compute_and_rank_sac_clusters(df_clusters, summary, min_cluster_size=5)

    
    
            # Process clusters 0 → 19
    clusters_to_process = list(range(0, 20))

    for cid in clusters_to_process:
        print(f"\n==============================")
        print(f" Processing cluster {cid}")
        print(f"==============================")

        # ---------------------------------------------
        # 1. Extract cluster data (normalized)
        # ---------------------------------------------
        cluster_df = cluster_data(df_clusters, summary, cid)
        
        cluster_df = extract_categorical_from_params(cluster_df)
        print(cluster_df["net_arch"].value_counts())
        print(cluster_df["ent_coef"].value_counts())

        # ---------------------------------------------
        # 2. Denormalize entire cluster
        # ---------------------------------------------


        cluster_df = denormalize_entire_cluster(
            log_path="paper2/sac",
            raw_csv_name="sac_studies.csv",
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
        compute_configs_from_cluster(out_csv)
    



    

    
    
   
   
    
    
    
    
    


if __name__ == "__main__":
    main()
