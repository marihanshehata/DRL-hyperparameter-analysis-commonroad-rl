import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fanova
from fanova import fANOVA
from fanova.visualizer import Visualizer
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
import optuna
from optuna.storages import RDBStorage
import copy
from optuna.trial import TrialState
import sqlalchemy as sa
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
sns.set(style="whitegrid")




# Extract data for FANOVA
def extract_data_from_study(study, config_space):
    """Extract hyperparameter configurations (X) and performance values (y) from the Optuna study."""
    trials = study.trials
    if len(trials) == 0:
        raise ValueError("No trials found in the Optuna study.")
    
    # Ensure correct order of hyperparameters
    param_names = [hp.name for hp in config_space.get_hyperparameters()]
    X = []
    y = []
    
    for trial in trials:
        # Extract hyperparameter values in the correct order
        X.append([trial.params.get(name, np.nan) for name in param_names])
        # Extract objective value
        y.append(trial.value)
    
    # Convert to NumPy arrays
    return np.array(X), np.array(y), param_names





def create_config_space(param_names, trials):
    """Create a ConfigSpace object from parameter names and trial data."""
    from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
    
     #params you want to exclude from FANOVA for SAC 
    EXCLUDED_PARAMS = {"net_arch","ent_coef","target_entropy"}


    cs = ConfigurationSpace()

    # Infer bounds from trials
    for name in param_names:
        if name in EXCLUDED_PARAMS:
            # skip net_arch
            continue
        values = [trial.params.get(name, None) for trial in trials if name in trial.params]
        if not values:
            raise ValueError(f"No values found for parameter {name} in trials.")

        # Remove None values and determine bounds
        values = [v for v in values if v is not None]
        min_val, max_val = min(values), max(values)

        # Add the hyperparameter to the ConfigSpace
        if all(isinstance(v, int) for v in values):
            cs.add_hyperparameter(UniformIntegerHyperparameter(name, min_val, max_val))
        elif all(isinstance(v, float) or isinstance(v, int) for v in values):
            cs.add_hyperparameter(UniformFloatHyperparameter(name, float(min_val), float(max_val)))
        else:
            
             # anything else (like strings, lists) will be ignored

             #raise ValueError(f"Unsupported type for parameter {name}: {values}")

            
            continue
       

    return cs




def verify_data(X, config_space):
    """Verify that all values in X fall within the bounds specified in config_space."""
    for i, hp in enumerate(config_space.get_hyperparameters()):
        min_val = hp.lower
        max_val = hp.upper
        column = X[:, i]

        if np.any(column < min_val) or np.any(column > max_val):
            print(f"Bad values for {hp.name}: {X[:, i][(X[:, i] < min_val) | (X[:, i] > max_val)]}")
            raise ValueError(f"Values for {hp.name} are outside the range [{min_val}, {max_val}].")

    print("Data verification passed: All values are within bounds.")


# Perform FANOVA analysis averaged over different seeds 

def perform_fanova_analysis(X, y, param_names, trials, log_path=None, seeds=[0, 42, 123, 999]):
    """Perform FANOVA analysis with averaging over multiple random seeds."""
    # Create ConfigSpace object
    config_space = create_config_space(param_names, trials)

    # Verify data integrity
    verify_data(X, config_space)

    # Filter out entries where y is None (e.g., pruned or failed trials)
    X_filtered = []
    y_filtered = []
    for xi, yi in zip(X, y):
        if yi is not None:
            X_filtered.append(xi)
            y_filtered.append(yi)

    X = np.array(X_filtered)
    y = np.array(y_filtered)



    # Initialize dictionaries to store cumulative results
    main_effects_cumulative = {param: [] for param in param_names}
    interaction_effects_cumulative = {}

    # Perform FANOVA multiple times with different seeds
    for seed in seeds:
        print(f"\nRunning FANOVA with seed={seed}")
        fanova = fANOVA(X, y, config_space=config_space, n_trees=100, seed=seed)
        
        # Calculate main effects for this seed
        for i, param in enumerate(param_names):
            importance = fanova.quantify_importance((i,))
            main_effects_cumulative[param].append(importance[(i,)]['individual importance'])
        
        # Calculate interaction effects (pairwise) for this seed
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                importance = fanova.quantify_importance((i, j))
                if (param_names[i], param_names[j]) not in interaction_effects_cumulative:
                    interaction_effects_cumulative[(param_names[i], param_names[j])] = []
                interaction_effects_cumulative[(param_names[i], param_names[j])].append(importance[(i, j)]['total importance'])

    # Average the results across seeds
    main_effects_avg = {param: np.mean(values) for param, values in main_effects_cumulative.items()}
    interaction_effects_avg = {pair: np.mean(values) for pair, values in interaction_effects_cumulative.items()}

    # Log averaged results
    if log_path:
        main_effects_path = os.path.join(log_path, "fanova_main_effects.txt")
        interaction_effects_path = os.path.join(log_path, "fanova_interaction_effects.txt")
        
        with open(main_effects_path, "w") as f:
            f.write("FANOVA Main Effects :\n")
            for param, importance in main_effects_avg.items():
                f.write(f"{param}: {importance:.4f}\n")
        
        with open(interaction_effects_path, "w") as f:
            f.write("FANOVA Interaction Effects:\n")
            for (param1, param2), importance in interaction_effects_avg.items():
                f.write(f"({param1}, {param2}): {importance:.4f}\n")

    print(f"\n results saved to {log_path}")
    return main_effects_avg, interaction_effects_avg




# Visualization
def visualize_fanova_results(main_effects, interaction_effects):
    """Visualize FANOVA main and interaction effects with enhanced plots."""
    sns.set_theme(style="whitegrid")  # Apply modern Seaborn theme
    
    # ---- Main Effects ----
    # Sort main effects for better visualization
    sorted_main_effects = dict(sorted(main_effects.items(), key=lambda x: x[1], reverse=True))
    hyperparameters = list(sorted_main_effects.keys())
    importances = list(sorted_main_effects.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=hyperparameters, palette="viridis")
    plt.xlabel("Importance Score", fontsize=14)
    plt.ylabel("Hyperparameters", fontsize=14)
    plt.title("FANOVA Main Effects", fontsize=16, pad=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("fanova_main_effects.png", dpi=300)
    plt.show()

    # ---- Interaction Effects ----
    # Prepare data for interaction heatmap
    params = sorted(set(param for pair in interaction_effects.keys() for param in pair))
    interaction_matrix = np.zeros((len(params), len(params)))

    for (param1, param2), importance in interaction_effects.items():
        i, j = params.index(param1), params.index(param2)
        interaction_matrix[i, j] = importance
        interaction_matrix[j, i] = importance  # Symmetric matrix

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        interaction_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=params,
        yticklabels=params,
        cbar_kws={"label": "Interaction Importance"},
        square=True
    )
    plt.title("FANOVA Interaction Effects Heatmap", fontsize=16, pad=20)
    plt.xticks(rotation=45, fontsize=12, ha="right")
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("fanova_interaction_effects.png", dpi=300)
    plt.show()






def plot_fanova_marginals(X, y, param_names,config_space , output_path="./ppo_fanova_marginals_plots", n_trees=100, seed=0,resolution=100, font_base=10):
    """
    Plot individual and pairwise partial dependence plots (PDPs) using fANOVA.

    Parameters:
    - X (array-like): Feature matrix of hyperparameters.
    - y (array-like): Objective values for the trials.
    - param_names (list of str): List of hyperparameter names.
    - output_path (str): Directory to save the plots.
    - n_trees (int): Number of trees for the random forest model.
    - seed (int): Random seed for reproducibility.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
     # Global font defaults (keeps it simple)
    plt.rcParams.update({
        "font.size": font_base,
        "axes.titlesize": font_base + 4,
        "axes.labelsize": font_base + 4 ,
        "xtick.labelsize": font_base,
        "ytick.labelsize": font_base,
        "legend.fontsize": font_base 
    })

    # Initialize fANOVA and Visualizer
    fanova = fANOVA(X, y, n_trees=n_trees, seed=seed)
    visualizer = Visualizer(fanova,config_space, output_path )




    # Plot individual marginals
    #print("Generating Individual Marginal plots...")
    #for i, param in enumerate(param_names):
     #   plt.figure()
     #   visualizer.plot_marginal(i,resolution=resolution)
     #   plt.savefig(os.path.join(output_path, f"{param}_pdp.png"))
     #   plt.close()
     #   print(f"single marginal for {param} saved.")

       
    # Plot pairwise marginals
    print("Generating Pairwise Marginal plots...")
    for i in range(len(param_names)):
        for j in range(i + 1, len(param_names)):
            plt.figure()
            visualizer.plot_pairwise_marginal([i, j])
         
            
            plt.savefig(os.path.join(output_path, f"{param_names[i]}_{param_names[j]}_pdp.png"))
            #plt.close()
            print(f"pairwise marginal for interaction between {param_names[i]} and {param_names[j]} saved.")
    
    
    
    
    print(f"All marginal plots saved in '{output_path}'.")
        

        
def compute_pdp_heatmap_averaged(X, y, param_names, config_space, output_path="fanova_pdp_plots", seeds=[0, 42, 123, 999], resolution=20):
    os.makedirs(output_path, exist_ok=True)

    print("Generating Pairwise PDP Heatmaps with Averaged Seeds...")
    
    for i in range(len(param_names)):
        for j in range(i + 1, len(param_names)):
            param1, param2 = param_names[i], param_names[j]

            # Define parameter ranges from ConfigSpace
            x_min, x_max = config_space[param1].lower, config_space[param1].upper
            y_min, y_max = config_space[param2].lower, config_space[param2].upper

            x_vals = np.linspace(x_min, x_max, resolution)
            y_vals = np.linspace(y_min, y_max, resolution)

            X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
            Z_vals_seeds = np.zeros((len(seeds), resolution, resolution))

            print(f"Processing PDP for {param1} vs {param2}...")

            # Run fANOVA for each seed and store results
            for s, seed in enumerate(seeds):
                fanova_instance = fANOVA(X, y, config_space=config_space, seed=seed)  # New fANOVA instance per seed
                
                for xi in range(resolution):
                    for yi in range(resolution):
                        mean, _ = fanova_instance.marginal_mean_variance_for_values(
                            (i, j), [X_mesh[yi, xi], Y_mesh[yi, xi]]
                        )
                        Z_vals_seeds[s, yi, xi] = mean  # Store mean values for this seed

            # Compute the average PDP values across all seeds
            Z_vals_avg = np.mean(Z_vals_seeds, axis=0)

            # Plot heatmap
            plt.figure(figsize=(8, 6))
            contour = plt.contourf(X_mesh, Y_mesh, Z_vals_avg, levels=20, cmap="plasma")
            plt.colorbar(contour, label="Averaged PDP Value")
            plt.xlabel(param1)
            plt.ylabel(param2)
            plt.title(f"Averaged PDP Heatmap: {param1} vs {param2}")
            plt.savefig(os.path.join(output_path, f"{param1}_{param2}_heatmap_avg.png"))
            plt.close()
            print(f"Saved Averaged PDP heatmap: {param1} vs {param2}")

    print(f"All PDP heatmaps with averaged seeds saved in '{output_path}'.")

    


def merge_studies(study_sources, merged_db_url, merged_study_name):
    """Merges trials from multiple studies."""
    
    
    
    # Ensure schema exists
    optuna.create_study(
        study_name="__init__",
        storage=merged_db_url,
        direction="minimize",
        load_if_exists=True
    )
    
    
    # Connect to the database manually
    engine = sa.create_engine(merged_db_url)

    # Try to delete the old study if it exists
    with engine.begin() as conn:
        # Find study id
        result = conn.execute(sa.text(f"SELECT study_id FROM studies WHERE study_name = :study_name"),
                              {"study_name": merged_study_name})
        row = result.fetchone()

        if row is not None:
            
            study_id = row[0]
            print(f"Found existing study '{merged_study_name}' with ID={study_id}. Deleting...")
            
            conn.execute(sa.text(f"DELETE FROM trial_intermediate_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = :study_id)"), {"study_id": study_id})
            
            conn.execute(sa.text(f"DELETE FROM trial_params WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = :study_id)"),{"study_id": study_id})
            
            conn.execute(sa.text(f"DELETE FROM trial_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = :study_id)"),{"study_id": study_id})
            
            conn.execute(sa.text(f"DELETE FROM trials WHERE study_id = :study_id"),{"study_id": study_id})
                        
           
            # NOW delete study-related metadata tables
            conn.execute(sa.text("DELETE FROM study_directions WHERE study_id = :study_id"), {"study_id": study_id})
        
        
        
            # Finally delete from studies
            conn.execute(sa.text("DELETE FROM studies WHERE study_id = :study_id"), {"study_id": study_id})
            
                    
            
            
            #conn.commit()
            print(f"Deleted study '{merged_study_name}'.")
        else:
            print(f"No existing study '{merged_study_name}' found.")

    # Now create a fresh study
    merged_study = optuna.create_study(
        study_name=merged_study_name,
        storage=merged_db_url,
        direction="minimize",
        load_if_exists=False,
    )

    # Merge trials from all studies
    for db_url, study_name in study_sources:
        study = optuna.load_study(study_name=study_name, storage=db_url)
        print(f" Copying from: {study_name}")

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                merged_study.add_trial(
                    optuna.trial.create_trial(
                        params=trial.params,
                        distributions=trial.distributions,
                        value=trial.value,
                        user_attrs=trial.user_attrs,
                        system_attrs=trial.system_attrs,
                        intermediate_values=trial.intermediate_values,
                        state=optuna.trial.TrialState.COMPLETE,
                    )
                )

    print(f"\nFinal merged study has {len(merged_study.trials)} complete trials.")
    
    
# Main function
def main():
   
    
    #using multi-studies (merged) to average results over seeds    
     
    #notice that for main and interaction effects use a separate code file that averages the resulted importances from each seed and compute mean and std with error bars
    
    
    
    #first we create a merged_study then proceed with it like normal      
    study_sources = [
    ("postgresql://postgres:pass@localhost/cs_seed0db", "study_8env_seed0"),
    ("postgresql://postgres:pass@localhost/cs_seed42db", "study_8env_seed42"),
    ("postgresql://postgres:pass@localhost/cs_seed123db", "study_8env_seed123"),
    ("postgresql://postgres:pass@localhost/cs_seed999db", "study_8env_seed999"),
    ("postgresql://postgres:pass@localhost/cs_seed2024db", "study_8env_seed2024"),]

    merged_db_url = "postgresql://postgres:pass@localhost/ppo_sparse_merged"
    merged_study_name = "ppo_sparse_merged_study"
   

    # This will create the required tables if they don't exist
    


    merge_studies(study_sources, merged_db_url, merged_study_name)

    #db_url = "postgresql://postgres:pass@localhost/sac_sparse_merged"  # pass is password here
    db_url= merged_db_url 
    storage = RDBStorage(db_url) 
        
    
    
    
    
    
    
    
    # Define paths of study pickle file  
    log_path='./logs'     
    
    #using one study directly    
    #load optuna study from database 
    #db_url = "postgresql://postgres:pass@localhost/sac_sparse_seed0"  # pass is password here
    #storage = RDBStorage(db_url)
    
  
    
    
    # List all studies (optional)
    for summary in storage.get_all_study_summaries():
        print("Found study:", summary.study_name)
    
    # Load your actual study by name
    #study = optuna.load_study(study_name="study", storage=storage)   
    
    
    study = optuna.load_study(study_name=merged_study_name, storage=storage)
    
    
    
    
    print("apply FANOVA on study : " ,study.study_name)
    
    
    trials = study.trials  # Pass trials for ConfigSpace creation
    
    # Create ConfigSpace object
    param_names = list(study.best_trial.params.keys())
    config_space = create_config_space(param_names, trials)
    
    # Extract data for FANOVA
    X, y, param_names = extract_data_from_study(study, config_space)
    
    
    # Verify data
    verify_data(X, config_space)  # Now X and config_space are both defined
    
    # Perform FANOVA analysis
    #main_effects, interaction_effects = perform_fanova_analysis(X, y, param_names, trials, log_path=log_path)
    
        
    # Visualize results
    #visualize_fanova_results(main_effects, interaction_effects)   
   


    
    #marginal plots   
    plot_fanova_marginals(X, y, param_names,config_space,resolution=50 )  


    # Initialize fANOVA
    #fanova_instance = fANOVA(X, y, config_space=config_space)
    
    #Fanova pdp
    #compute_pdp_heatmap_averaged(X, y, param_names, config_space)
    
    

    
    
    

if __name__ == "__main__":
    main()
