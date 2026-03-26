import argparse
import os
import pandas as pd
import numpy as np
import ast


# ============================================================
# RANGE-SNAPPING UTILITIES (only for categorical HP)
# ============================================================

def snap_to_range_generic(key, val, param_ranges):
    allowed = param_ranges.get(key)
    if allowed is None:
        return val
    if isinstance(allowed, list):
        return min(allowed, key=lambda x: abs(x - val))
    return val


def snap_to_range_sac(key, val):
    param_ranges = {
        "gamma":        [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        "batch_size":   [16, 32, 64, 128, 256, 512],
        "buffer_size":  [int(1e4), int(1e5), int(1e6)],
        "gradient_steps": [1, 10, 100, 300],
    }
    return snap_to_range_generic(key, val, param_ranges)


def snap_to_range_ppo(key, val):
    param_ranges = {
        "batch_size":  [8, 16, 32, 64, 128, 256, 512],
        "n_steps":     [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        "gamma":       [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        "cliprange":   [0.1, 0.2, 0.3, 0.4],
        "noptepochs":  [1, 5, 10, 20, 30, 50],
        "lamdba":      [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0],
    }
    return snap_to_range_generic(key, val, param_ranges)


# ============================================================
# CLUSTER HP STATISTICS
# ============================================================

def compute_cluster_stats(df, algo):
    exclude_cols = ["cluster", "params", "seed", "value", "reward", "trial_number", "target_entropy"]
    hp_cols = [c for c in df.columns if c not in exclude_cols]

    if algo == "ppo":
        categorical_hps = {"batch_size", "n_steps", "gamma", "cliprange", "noptepochs", "lamdba"}
    else:
        categorical_hps = {"gamma", "batch_size", "buffer_size", "gradient_steps", "ent_coef", "net_arch"}

    # Snapping categorical values
    for hp in hp_cols:
        if hp in categorical_hps:
            snapped = []
            for v in df[hp]:
                if algo == "ppo":
                    snapped.append(snap_to_range_ppo(hp, v))
                else:
                    snapped.append(snap_to_range_sac(hp, v))
            df[hp] = snapped

    full_stats = []

    for hp in hp_cols:
        col = df[hp]

        if hp in categorical_hps:
            col_str = col.dropna().astype(str)
            if col_str.empty:
                full_stats.append({
                    "hp": hp, "type": "categorical",
                    "mode": None, "entropy": None
                })
                continue

            vc = col_str.value_counts()
            total = len(col_str)
            pct = (vc / total * 100)

            prob = vc / total
            entropy = -(prob * np.log2(prob)).sum()

            row = {"hp": hp, "type": "categorical",
                   "mode": vc.index[0],
                   "entropy": round(float(entropy), 4)}

            for cat, val in pct.items():
                row[f"pct_{cat}"] = round(val, 2)

            full_stats.append(row)

        else:
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)

            full_stats.append({
                "hp": hp,
                "type": "numeric",
                "median": col.median(),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "min": col.min(),
                "max": col.max(),
            })

    return full_stats


# ============================================================
# MAIN PIPELINE — LOOP OVER CLUSTERS, SAVE CSV + EXCEL
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"], required=True)
    parser.add_argument("--path", type=str, default="paper2")
    args = parser.parse_args()

    algo = args.algo
    base = args.path

    summaries = []

    for i in range(20):
        file_path = os.path.join(base, algo, f"cluster_{i}_data.csv")

        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # SAC: parse params dict
        if algo == "sac" and "params" in df.columns:
            try:
                parsed = df["params"].apply(ast.literal_eval)
                df["ent_coef"] = parsed.apply(lambda x: x.get("ent_coef", None))
                df["net_arch"] = parsed.apply(lambda x: x.get("net_arch", None))
            except:
                print("Warning: cannot parse params column")

        cluster_stats = compute_cluster_stats(df, algo)

        for row in cluster_stats:
            row["cluster"] = i
            summaries.append(row)

    summary_df = pd.DataFrame(summaries)

    out_folder = os.path.join(base, algo, "hp_summaries")
    os.makedirs(out_folder, exist_ok=True)

     # =======================================================
    # Save master summary with algo name
    # =======================================================

    master_csv = os.path.join(out_folder, f"{algo}_all_clusters_hp_summary.csv")
    summary_df.to_csv(master_csv, index=False)

    master_excel = os.path.join(out_folder, f"{algo}_all_clusters_hp_summary.xlsx")
    with pd.ExcelWriter(master_excel, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="all_hp")

    print(f"\nSaved master summary:\n{master_csv}\n{master_excel}")

    # =======================================================
    # Phase 2A — Split into numeric and categorical tables
    # =======================================================

    numeric_df = summary_df[summary_df["type"] == "numeric"].copy()
    categorical_df = summary_df[summary_df["type"] == "categorical"].copy()

    # ---------- Save numeric summary ----------
    numeric_csv = os.path.join(out_folder, f"{algo}_numeric_summary.csv")
    numeric_df.to_csv(numeric_csv, index=False)

    numeric_excel = os.path.join(out_folder, f"{algo}_numeric_summary.xlsx")
    with pd.ExcelWriter(numeric_excel, engine="openpyxl") as writer:
        numeric_df.to_excel(writer, index=False, sheet_name="numeric")

    # ---------- Save categorical summary ----------
    categorical_csv = os.path.join(out_folder, f"{algo}_categorical_summary.csv")
    categorical_df.to_csv(categorical_csv, index=False)

    categorical_excel = os.path.join(out_folder, f"{algo}_categorical_summary.xlsx")
    with pd.ExcelWriter(categorical_excel, engine="openpyxl") as writer:
        categorical_df.to_excel(writer, index=False, sheet_name="categorical")

    print("\nPhase 2A complete:")
    print(f"Numeric summary saved to:\n  {numeric_csv}\n  {numeric_excel}")
    print(f"Categorical summary saved to:\n  {categorical_csv}\n  {categorical_excel}")
if __name__ == "__main__":
    main()

