## =====================================================================
## Objective: Train a SOM using benthic cover (not per site) to uncover gradients in composition space
## Output: U-Matrix, component planes, node clustering (KMeans)
## Input features: benthic cover data
## =====================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Patch, Wedge
from matplotlib import cm, colorbar
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict, Counter
import pickle
import argparse
import math
import sys
import random

def load_data(filepath):
    df = pd.read_csv(filepath)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# ----------------------------------------------------------------------
# base path
#----------------------------------------------------------------------
base_paths = {
    "driver": "/home/bernard/cbmserino/IMEE/marine_som",
    "pc": "/home/bernard/Documents/class3/marine_som",
    "dropbox": "/home/bernard/Dropbox/IMEE/marine_som",
}
files = {
    "df1": ["final_dataset", "clustered_dataset.csv"],
    "df2": ["final_dataset", "cc.csv"],
    "df3": ["final_dataset", "labels.csv"]
}
    
# -- load dataset 'cover&chem.csv'--
def load_dataset(base_path, relative_path):
    full_path = os.path.join(base_path, *relative_path)
    return pd.read_csv(full_path)
    
df1 = load_dataset(base_paths["dropbox"], files["df1"])
df2 = load_dataset(base_paths["dropbox"], files["df2"])
df3 = load_dataset(base_paths["dropbox"], files["df3"])

# ----------------------------------------------------------------------
# result save path
#-----------------------------------------------------------------------
save_paths = [
    "/home/bernard/Documents/class3/marine_som_results/",
    "/home/bernard/Dropbox/IMEE/marine_som",
    "/home/bernard/cbmserino/IMEE/marine_som"
]

def make_save_dirs(save_paths, 
                   phase="1phase", 
                   iterations=500, 
                   dimension="11x11",
                   model_name="cosine_som_sp11x11"):
    final_paths = []    
    for base in save_paths:
        try:
            subfolder = os.path.join(
                base,
                phase,
                f"{iterations}iter",
                dimension,
                model_name
            )
            os.makedirs(subfolder, exist_ok=True)
            final_paths.append(subfolder)
            print(f"Created directory: {subfolder}")
        except Exception as e:
            print(f"Failed to create {base}: {str(e)}")
    
    return final_paths if final_paths else None

result_dirs = make_save_dirs(
    save_paths,
    phase="2phase",
    iterations=144000,
    dimension="29x27",
    model_name="cosine_som_sp"
)
print("Available save directories:", result_dirs)


# ----------------------------------------------------------------------
# feature selection
# ----------------------------------------------------------------------
'''define groups'''
env_vars = ['Mean_SST', 'SD_SST', 'Light_intensity', 'Wave_exposure', 'Wave_height', 'Mean_chl_a', 'SD_chl_a', 'Nitrate', 'Nitrite', 'Phosphate', 'DHW', 'DHW_recovery', 'Typhoon_disturbance', 'Typhoon_recovery', 'Typhoon_frequency', 'Anthropogenic_land_use', 'Forest_land_use', 'Population_density', 'Tourist_visitors', 'Unstable_substrate_cover']

# ~ chem_vars = ['Cd (mg/l)', 'Chl a (μg/l)', 'Cu (mg/l)', 'DO (mg/l)', 'DOsat (%)', 
                # ~ 'Hg (mg/l)', 'NH3 (mg/l)', 'NO2 (mg/l)', 'NO3 (mg/l)', 'PO4 (mg/l)', 
                # ~ 'Pb (mg/l)', 'SS (mg/l)', 'Sal (psu)', 'Sio4 (mg/l)', 'T (℃)', 
                # ~ 'WT (℃)', 'Zn (mg/l)', 'pH']
chem_vars = [
    'Cd (mg/L)', 'Chl a (μg/L)', 'Cu (mg/L)', 'DO (mg/L)', 'DOsat (%)', 
    'Hg (mg/L)', 'NH3 (mg/L)', 'NO2 (mg/L)', 'NO3 (mg/L)', 'PO4 (mg/L)', 
    'Pb (mg/L)', 'SS (mg/L)', 'Sal (psu)', 'SiO4 (mg/L)', 'T (℃)', 
    'WT (℃)', 'Zn (mg/L)', 'pH']

major_cats = ['Actiniaria', 'Artificial Debris', 'Ascidian', 'Black coral', 'CCA', 'Corallimorpharia', 'Fish', 'Gorgonian', 'Hard coral', 'Hydrozoa', 'Macroalgae', 'Other mobile invertebrates', 'Other sessile invertebrates', 'Seagrass', 'Soft coral', 'Sponge', 'Stable substrate', 'TWS', 'Turf', 'Unstable substrate', 'Zoanthid']

nonbio_vars = ['fistws', 'nettws', 'shatws', 'taptws', 'unktws', 'wantws']

kmeans_pca = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "cluster"]

abiotic = ['boulss', 'gravus', 'limess', 'rockss', 'rubbus', 'sandus', 'siltus']

sp_vars = [col for col in df1.columns if col not in chem_vars + major_cats + kmeans_pca + nonbio_vars + abiotic + ['ochoth', 'year', 'site', 'region', 'SD', 'depth', 'transect', 'nlabels', 'latitude', 'longitude', 'Shannon', 'Pielou', 'BC']]


# features to train
variables_of_interest = sp_vars

'''
# ----------------------------------------------------------------------
# Data preprocessing
# ----------------------------------------------------------------------
def preprocess_data(df, features):
    X = df[features].fillna(0)
    
    # log-transform skewed variables
    for var in X.columns:
        if X[var].min() >= 0:  # log-transform non-negative variables
            X[var] = np.log1p(X[var])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

# -------------------------+
# scale feature to train   |
# -------------------------+
X_scaled, scaler = preprocess_data(df1, variables_of_interest)'''

# ---------------------------------------------------------------------+
# Hellinger transformation                                             |
# ---------------------------------------------------------------------+
def hellinger_transform(df, features):
    X = df[features].fillna(0).values.astype(float)
    
    # relative abundance
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    X_rel = X / row_sums
    
    # Hellinger (square root of relative abundance)
    X_hel = np.sqrt(X_rel)
    
    return X_hel

# -------------------------+
# transform features       |
# -------------------------+
X_hel = hellinger_transform(df2, variables_of_interest)

# --------------------+
# Optimized dimension | 
# --------------------+
def recommend_iterations(data_size, som_size):
    """Recommend iterations based on data and SOM size"""
    base_iterations = data_size * 2  # 2 epochs minimum
    
    if som_size <= 10:  # small SOM
        return min(base_iterations, 1000)
    elif som_size <= 20:  # Medium SOM
        return min(base_iterations, 2000)
    else:  # Large SOM
        return min(base_iterations, 5000)
        
data_size = len(X_hel)
som_size = 20  # dimenstion
recommended = recommend_iterations(data_size, som_size)
print(f"Recommended iterations: {recommended}")

# ----------------------------------------------------------------------
# save trained SOM
# ----------------------------------------------------------------------
def save_som_data(som, X_hel, result_dirs):
    if not result_dirs:
        print("No valid result directories provided.")
        return {}

    saved_files = {}
    bmu_locations = np.array([som.winner(x) for x in X_hel])
    som_weights = som.get_weights()

    for result_dir in result_dirs:
        try:
            os.makedirs(result_dir, exist_ok=True)
            paths = {
                "bmu_locations": os.path.join(result_dir, "bmu_locations.npy"),
                "X_hel": os.path.join(result_dir, "X_hel.npy"),
                "som_weights": os.path.join(result_dir, "som_weights.npy")
            }

            np.save(paths["bmu_locations"], bmu_locations)
            np.save(paths["X_hel"], X_hel)
            np.save(paths["som_weights"], som_weights)

            for name, path in paths.items():
                if os.path.isfile(path):
                    saved_files[name] = path
                    print(f"Saved {name} to: {path}")
                else:
                    print(f"Failed to save {name} to: {path}")
        except Exception as e:
            print(f"Error saving SOM data to {result_dir}: {e}")
    
    return saved_files
    
'''
# ----------------------------------------------------------------------
# Training (one phase)
# ----------------------------------------------------------------------
np.random.seed(42)
random.seed(42)

som_dim = 20
som = MiniSom(x=som_dim, y=som_dim, input_len=X_hel.shape[1], 
              sigma=3.0, learning_rate=0.5, 
              neighborhood_function='gaussian', 
              random_seed=42, 
              activation_distance='cosine')

# Batch training 
# ~ som.train_batch(X_hel, num_iterations=1392, verbose=True) 

# Manual training for QE tracking
sample_order = np.random.permutation(len(X_hel))  # fixed permutation    
quantization_errors = []
num_iterations = 1392

for i in tqdm(range(num_iterations), desc="Training SOM"): 
    sample_idx = sample_order[i % len(X_hel)]  # deterministic sampling
    # ~ sample_idx = i % len(X_hel) # sequential, deterministic
    som.update(X_hel[sample_idx], som.winner(X_hel[sample_idx]), i, num_iterations)
    
    if i % 10 == 0:
        qe = np.mean([np.linalg.norm(x - som.get_weights()[som.winner(x)]) for x in X_hel])
        quantization_errors.append(qe)

print(f"Final quantization error: {quantization_errors[-1]:.4f}")
print("ONE PHASE: Training complete (cosine SOM).")

# Verification test: consistency (for manual training)
weights_run1 = som.get_weights().copy() # retraining (twice) to compare final weights
weights_run2 = som.get_weights().copy() # reset and retrain
print(f"Weights identical: {np.array_equal(weights_run1, weights_run2)}") # compare final weights
'''
# ----------------------------------------------------------------------
# Training (Vesanto & Alhoniemi (2000), two-phase)
# ----------------------------------------------------------------------
som_dim = (29,27)
input_len = X_hel.shape[1]

som = MiniSom(
    x=som_dim[0],
    y=som_dim[1],
    input_len=input_len,
    sigma=3.0,                     # large neighborhood for rough phase
    learning_rate=0.5,             # high learning rate for rough phase
    neighborhood_function='gaussian',
    random_seed=42,
    activation_distance='cosine' # distance metric
)

# training schedule parameters
num_iterations_rough = 3 * len(X_hel)      # rough phase (≈ 3 epochs)
num_iterations_fine = 10 * len(X_hel)      # fine-tuning phase (≈ 10 epochs)

# rough phase
for i in tqdm(range(num_iterations_rough), desc="Rough training"):
    x = X_hel[np.random.randint(0, len(X_hel))]
    som.update(x, som.winner(x), i, num_iterations_rough)

# adjust parameters for fine-tuning
som.sigma = 1.0                              # narrower neighborhood for fine tuning
som.learning_rate = 0.05                     # smaller step size for fine tuning

# fine-tuning phase
for i in tqdm(range(num_iterations_fine), desc="Fine-tuning"):
    x = X_hel[np.random.randint(0, len(X_hel))]
    som.update(x, som.winner(x), i, num_iterations_fine)

# compute quantization error
quantization_error = np.mean([np.linalg.norm(x - som.get_weights()[som.winner(x)]) for x in X_hel])
print(f"Final quantization error: {quantization_error:.4f}")
print("TWO PHASE: Training complete (Vesanto & Alhoniemi, 2000).")

save_som_data(som, X_hel, result_dirs)

# ----------------------------------------------------------------------
# Save BMU locations and scaled data
# ----------------------------------------------------------------------
def save_som_data_cosine(som, X_hel, result_dirs):
    if not result_dirs:
        print("No valid result directories provided.")
        return {}

    saved_files = {}
    bmu_locations = np.array([som.winner(x) for x in X_hel])
    som_weights = som.get_weights()

    for result_dir in result_dirs:
        try:
            os.makedirs(result_dir, exist_ok=True)
            paths = {
                "bmu_locations": os.path.join(result_dir, "bmu_locations.npy"),
                "X_hel": os.path.join(result_dir, "X_hel.npy"),
                "som_weights": os.path.join(result_dir, "som_weights.npy")
            }

            np.save(paths["bmu_locations"], bmu_locations)
            np.save(paths["X_hel"], X_hel)
            np.save(paths["som_weights"], som_weights)

            for name, path in paths.items():
                if os.path.isfile(path):
                    saved_files[name] = path
                    print(f"Saved {name} to: {path}")
                else:
                    print(f"Failed to save {name} to: {path}")
        except Exception as e:
            print(f"Error saving SOM data to {result_dir}: {e}")
    
    return saved_files

save_som_data_cosine(som, X_hel, result_dirs)

# ----------------------------------------------------------------------
# Save model
# ----------------------------------------------------------------------
for path in result_dirs:
    try:
        model_save_path = os.path.join(path, "cosine_sp_20x20.pkl")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        with open(model_save_path, 'wb') as f:
            pickle.dump(som, f)
        print(f"✅ Trained SOM model saved to {model_save_path}")
        
        if os.path.exists(model_save_path):
            print("   Model file verification: success")
        else:
            print("   ⚠️: Model file not found after saving!")
    except Exception as e:
        print(f"❌ Error saving model to {path}: {e}")

# ----------------------------------------------------------------------
# Load saved BMU locations and scaled data for species overlay
# ----------------------------------------------------------------------
try:
    bmu_locations = np.load(os.path.join(result_dirs[0], "bmu_locations.npy"))
except Exception as e:
    print(f"Error loading BMU locations: {str(e)}")
    bmu_locations = np.array([som.winner(x) for x in X_hel])

try:
    X_hel = np.load(os.path.join(result_dirs[0], "X_hel.npy"))
except Exception as e:
    print(f"Error loading scaled data: {str(e)}")

try:
    bmu_locations = np.load(os.path.join(result_dirs[0], "bmu_locations.npy"))
except Exception as e:
    print(f"Error loading BMU locations: {str(e)}")
    bmu_locations = np.array([som.som.winner(x) for x in X_hel])

try:
    X_hel = np.load(os.path.join(result_dirs[0], "X_hel.npy"))
except Exception as e:
    print(f"Error loading scaled data: {str(e)}")
    
# ----------------------------------------------------------------------
# Compute BMU-wise averages
# ----------------------------------------------------------------------
def compute_bmu_averages(df, X_hel, som, sp_vars):
    n_nodes_x, n_nodes_y = som._weights.shape[:2]
    bmu_species = np.zeros((n_nodes_x, n_nodes_y, len(sp_vars)))
    # ~ bmu_envs = np.zeros((n_nodes_x, n_nodes_y, len(env_vars)))
    bmu_counts = np.zeros((n_nodes_x, n_nodes_y))

    for idx in range(len(df)):
        x, y = som.winner(X_hel[idx])
        bmu_species[x, y] += df.loc[idx, sp_vars].fillna(0).values
        # ~ bmu_envs[x, y] += df.loc[idx, env_vars].fillna(0).values
        bmu_counts[x, y] += 1

    # Avoid division by zero
    bmu_species_avg = np.divide(
        bmu_species, bmu_counts[..., np.newaxis], 
        out=np.zeros_like(bmu_species), 
        where=bmu_counts[..., np.newaxis] != 0
    )
    # ~ bmu_envs_avg = np.divide(
        # ~ bmu_envs, bmu_counts[..., np.newaxis], 
        # ~ out=np.zeros_like(bmu_envs), 
        # ~ where=bmu_counts[..., np.newaxis] != 0
    # ~ )

    return bmu_species_avg

bmu_sp_avg = compute_bmu_averages(df1, X_hel, som, sp_vars)

# ----------------------------------------------------------------------
# Optimal Number of KMeans Clusters for SOM Nodes - WCSS (Elbow), Silhouette, and DBI for SOM node clustering across K=2–15
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Flatten SOM weights (each node as observation)
weights = som.get_weights().reshape(-1, som.get_weights().shape[-1])

# Range of cluster numbers
K = range(2, 15)
wcss, silhouette, dbi = [], [], []

print("Evaluating KMeans cluster quality across K=2–15 ...")

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(weights)
    
    # Within-cluster sum of squares
    wcss.append(kmeans.inertia_)
    
    # Silhouette score
    sil = silhouette_score(weights, labels)
    silhouette.append(sil)
    
    # Davies-Bouldin index
    db = davies_bouldin_score(weights, labels)
    dbi.append(db)
    
    print(f"K={k:2d} | WCSS={kmeans.inertia_:.2f} | Silhouette={sil:.3f} | DBI={db:.3f}")

# Plot all metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Elbow Method
axes[0].plot(K, wcss, 'o-', color='tab:blue')
axes[0].set_title('Elbow Method (WCSS)')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('WCSS (lower is better)')

# 2. Silhouette Coefficient
axes[1].plot(K, silhouette, 'o-', color='tab:green')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Mean Silhouette (higher is better)')

# 3. Davies–Bouldin Index
axes[2].plot(K, dbi, 'o-', color='tab:red')
axes[2].set_title('Davies–Bouldin Index')
axes[2].set_xlabel('Number of Clusters (k)')
axes[2].set_ylabel('DBI (lower is better)')

plt.tight_layout()

# Determine suggested K based on criteria

opt_sil = K[np.argmax(silhouette)]
opt_dbi = K[np.argmin(dbi)]

print(f"\n🔹 Suggested number of clusters based on Silhouette: {opt_sil}")
print(f"🔹 Suggested number of clusters based on DBI: {opt_dbi}")

results_df = pd.DataFrame({
    "K": K,
    "WCSS": wcss,
    "Silhouette": silhouette,
    "DaviesBouldin": dbi
})

for path in result_dirs:
    csv_path = os.path.join(path, "kmeans_cluster_quality_summary.csv")
    fig_path = os.path.join(path, "kmeans_optimal_k_plots.png")
    results_df.to_csv(csv_path, index=False)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics table and plots to: {path}")

plt.show()

# ----------------------------------------------------------------------    
# Save all training output
# ----------------------------------------------------------------------
# export dictionary
export_data = {
    'som': som,
    'X_hel': X_hel,
    'df': df1,
    'variables_of_interest': variables_of_interest,
    'bmu_sp_avg': bmu_sp_avg,
    # ~ 'bmu_env_avg': bmu_env_avg,
    'bmu_locations': bmu_locations
    # ~ 'node_clusters': node_clusters
}

for path in result_dirs:
    try:
        model_save_path = os.path.join(path, "som_training_output.pkl")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        with open(model_save_path, 'wb') as f:
            pickle.dump(export_data, f)
        print(f"✅ Trained SOM model saved to {model_save_path}")
        
        if os.path.exists(model_save_path):
            print("   Model file verification: success")
        else:
            print("   ⚠️: Model file not found after saving!")
    except Exception as e:
        print(f"❌ Error saving model to {path}: {e}")

print("..... \n.... \n... \n.. \n. \n \nTraining complete!")        
        
# ----------------------------------------------------------------------
# Quantization error training visualization
# ----------------------------------------------------------------------

def plot_training_progress(errors):
    """plot quantization error over training iterations"""
    plt.figure(figsize=(10, 5))
    plt.plot(errors)
    plt.title('cosine distance-based SOM Training Progress (Quantization Error)')
    plt.xlabel('Iteration')
    plt.ylabel('Average Distance to BMU')
    plt.grid(True)
    for path in result_dirs:
        try:
            '''check if directory exists'''
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, "training_progress.png")
            plt.savefig(file_path, dpi=300)
            print(f"Plot saved successfully to: {file_path}")
        except Exception as e:
            print(f"Failed to save plot to {path}: {e}")
    plt.show()

# ~ plot_training_progress(quantization_errors)

print("..... \n.... \n... \n.. \n.")
