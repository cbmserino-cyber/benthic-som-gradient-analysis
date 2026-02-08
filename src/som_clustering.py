## =====================================================================
## Objective: Analyze trained SOM using benthic cover (not per site) to uncover gradients in composition space and link them to environmental gradient
## Output: kmeans on node weights + assign BMU clusters, node clustering (KMeans)
## Input features: benthic cover data
## =====================================================================

import sys
from datetime import datetime
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
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import pickle, argparse, math, sys, os
from shapely.geometry import Point
import matplotlib
matplotlib.use('Agg')
from shapely.geometry import Point
from scipy.stats import entropy


# ----------------------------------------------------------------------
# Log output/results
# ----------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"analysis_minisom_base_v2_{timestamp}.txt"

class DualOutput:
    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()

# Open log file and redirect output
log_file = open(log_filename, "w", encoding="utf-8")
original_stdout = sys.stdout
sys.stdout = DualOutput(sys.stdout, log_file)


# ----------------------------------------------------------------------
# base path
#----------------------------------------------------------------------
base_paths = {
    "driver": "/home/bernard/cbmserino/IMEE/marine_som",
    # ~ "pc": "/home/bernard/Documents/class3/marine_som",
    "dropbox": "/home/bernard/Dropbox/IMEE/marine_som"
}
files = {
    "df1": ["final_dataset", "cc.csv"],
    "df2": ["final_dataset", "clustered_dataset.csv"],
    "df3": ["final_dataset", "labels.csv"]
}
    
# -- load dataset 'cover&chem.csv'--
def load_dataset(base_path, relative_path):
    full_path = os.path.join(base_path, *relative_path)
    return pd.read_csv(full_path)
    
df1 = load_dataset(base_paths["dropbox"], files["df1"])
# ~ df2 = load_dataset(base_paths["dropbox"], files["df2"])
df3 = load_dataset(base_paths["dropbox"], files["df3"])

save_paths = [
    # ~ "/home/bernard/Documents/class3/marine_som_results/",
    "/home/bernard/Dropbox/IMEE/marine_som",
    "/home/bernard/cbmserino/IMEE/marine_som"
]

def make_save_dirs(save_paths, 
                   phase="1phase", 
                   iterations=500, 
                   dimension="11x11",
                   model_name="manhattan_som_sp11x11"):
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
    model_name="manhattan_som_sp"
)
print("Available base directories:", result_dirs)

# ----------------------------------------------------------------------
# Analysis path with n_clusters
# ----------------------------------------------------------------------
n_clusters=5
def make_save_dirs(save_paths, 
                   phase="1phase", 
                   iterations=500, 
                   dimension="11x11",
                   model_name="manhattan_som_sp11x11",
                   n_clusters=5):
    final_paths = []    
    for base in save_paths:
        try:
            subfolder = os.path.join(
                base,
                phase,
                f"{iterations}iter",
                dimension,
                model_name,
                f"nclust{n_clusters}" 
            )
            os.makedirs(subfolder, exist_ok=True)
            final_paths.append(subfolder)
            print(f"Created directory: {subfolder}")
        except Exception as e:
            print(f"Failed to create {base}: {str(e)}")
    
    return final_paths if final_paths else None

analysis_dirs = make_save_dirs(
    save_paths,
    phase="2phase",
    iterations=144000,
    dimension="29x27",
    model_name="manhattan_som_sp",
    n_clusters=5
)
print("Available analysis directories:", analysis_dirs)

# ----------------------------------------------------------------------
# Feature selection
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


'''select features to train'''
# ----------------------------
variables_of_interest = sp_vars

# ----------------------------------------------------------------------
# Load saved SOM model
# ----------------------------------------------------------------------
output_path = os.path.join(result_dirs[0], "som_training_output.pkl")
def load_training_output(output_path):
    with open(output_path, 'rb') as f:
        data = pickle.load(f)
    
    # list required keys from trained model
    required_keys = ['som', 'X_hel', 'df', 'variables_of_interest',
                   'bmu_sp_avg']
    
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key in training output: {key}")
    
    return data

try:
    training_data = load_training_output(output_path)
    
    # unpack training output
    som = training_data['som']
    X_hel = training_data['X_hel']
    df2 = training_data['df']
    variables_of_interest = training_data['variables_of_interest']
    bmu_sp_avg = training_data['bmu_sp_avg']
    # ~ bmu_env_avg = training_data['bmu_env_avg']
    # ~ node_predicted_cluster_map = training_data['node_predicted_cluster_map']
    bmu_locations = training_data['bmu_locations']
    
except Exception as e:
    print(f"Error loading training output: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------
# Load saved BMU locations and scaled data for species overlay
# ----------------------------------------------------------------------
try:
    bmu_locations = np.load(os.path.join(result_dirs[1], "bmu_locations.npy"))
except Exception as e:
    print(f"Error loading BMU locations: {str(e)}")
    bmu_locations = np.array([som.winner(x) for x in X_hel])

try:
    X_hel = np.load(os.path.join(result_dirs[0], "X_hel.npy"))
except Exception as e:
    print(f"Error loading scaled data: {str(e)}")

weights = som.get_weights()
xdim, ydim, n_features = weights.shape
umatrix = som.distance_map()
som_shape = som.get_weights().shape[:2]
xrange = np.arange(som_shape[0])
yrange = np.arange(som_shape[1])
xx, yy = np.meshgrid(xrange, yrange)
xx, yy = xx, yy

# ----------------------------------------------------------------------
# Predict SOM cluster from KMeans model
# ----------------------------------------------------------------------

# kmeans model base bath:
base_path = "/home/bernard/Dropbox/IMEE/marine_som/kmeans_models"
model_path = os.path.join(base_path, "kmeans_model.pkl")

with open(model_path, "rb") as f:
    kmeans_model = pickle.load(f)

# 1.2. reshape to 2D array
weights_reshaped = weights.reshape(xdim * ydim, n_features) 

# execute: predict cluster label per node
node_labels = kmeans_model.predict(weights_reshaped) + 1

# 1.3. reshape back to SOM map grid
node_predicted_cluster_map = node_labels.reshape(xdim, ydim)

print("SOM Node Cluster Map (Predicted):")
print(node_predicted_cluster_map)

# 1.4. map predicted BMU cluster to its matching input vectors
def assign_clusters(som_obj, data, node_predicted_cluster_map):
    assignments = np.zeros(len(data), dtype=int)
    
    for i, x in enumerate(data):
        winner = som_obj.winner(x)
        assignments[i] = node_predicted_cluster_map[winner[0], winner[1]]
    
    return assignments

# execute: assign new column for cluster label per input
df2['predicted_Cluster'] = assign_clusters(som, X_hel, node_predicted_cluster_map)

# 1.5. save new dataframe
for path in analysis_dirs:
    try:
        final_path = os.path.join(path, "df2_predicted_clusters.csv")
        df2.to_csv(final_path, index=False)
        print(f"Final DataFrame with SOM clusters saved to: {final_path}")
    except Exception as e:
        print(f"Failed to save final DataFrame to {path}: {e}")

# 2.1compute transition entropy for each node - how mixed the samples assigned to that node are)
bmu_assignments = defaultdict(list)
labels_raw = df2['predicted_Cluster'].values   # sample labels from dataset

# 2.2. assign cluster label to BMU
for i, x in enumerate(X_hel):
    node = som.winner(x)
    bmu_assignments[node].append(labels_raw[i])

node_entropy = np.zeros((xdim, ydim))

# 2.3. execute: entropy
for (x, y), cluster_list in bmu_assignments.items():
    counts = np.bincount(cluster_list)
    probs = counts[counts > 0] / counts.sum()
    node_entropy[x, y] = entropy(probs)  # Shannon entropy

# 3.1. create discrete colormap for clusters
n_predicted_clusters = kmeans_model.n_clusters

# 3.2. get all unique cluster IDs from the map
cluster_ids = np.unique(node_predicted_cluster_map)

# 3.3. kmeans cluster color assignment:
tab20 = plt.get_cmap('tab20')
custom_indices = [0, 5, 10, 15, 19, 14]

if n_predicted_clusters > len(custom_indices):
    # If more clusters than custom colors, extend with more indices
    print(f"Warning: Need {n_predicted_clusters} colors but only {len(custom_indices)} provided.")
    print("Extending with additional colors from tab20...")
    # Add more indices to cover all clusters
    all_indices = list(range(20))  # tab20 has 20 distinct colors
    # Remove duplicates from custom_indices first
    remaining = [i for i in all_indices if i not in custom_indices]
    custom_indices = custom_indices + remaining[:n_predicted_clusters - len(custom_indices)]

selected_indices = custom_indices[:n_predicted_clusters]
discrete_color = [tab20(i) for i in selected_indices]
discrete_tab20 = ListedColormap(discrete_color)  # usage: discrete_tab20() i.e. cluster_color(i)/(c1)/(c2)

# 3.4. create color mapping for each cluster
sorted_clusters = sorted(cluster_ids)
cluster_colors = {
    cluster: discrete_color[i] for i, cluster in enumerate(sorted_clusters)
}

print(f"\nColor assignments:")
for cluster, color in cluster_colors.items():
    rgb = [int(c * 255) for c in color[:3]]
    print(f"  Cluster {cluster}: RGB {rgb}")

plt.figure(figsize=(12, 12))

# 3.5. create the heatmap
plt.imshow(node_predicted_cluster_map, 
           cmap=discrete_tab20, 
           origin='lower',  # Same as U-Matrix
           aspect='auto')

# Add annotations
for i in range(xdim):
    for j in range(ydim):
        plt.text(j, i, str(node_predicted_cluster_map[i, j]),
                 ha='center', va='center',
                 color='black' if node_predicted_cluster_map[i, j] % 2 == 0 else 'black')

# ~ plt.colorbar(label='Cluster')
plt.title('SOM Node Clusters (Predicted by KMeans)')
# ~ plt.xlabel('SOM X Dimension')
# ~ plt.ylabel('SOM Y Dimension')

# Add grid coordinates for reference (matching U-Matrix)
# ~ plt.xticks(np.arange(node_predicted_cluster_map.shape[1]))
# ~ plt.yticks(np.arange(node_predicted_cluster_map.shape[0]))

for path in analysis_dirs:
    try:
        os.makedirs(path, exist_ok=True)
        
        file_path = os.path.join(path, "som_node_predicted_cluster_map.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved successfully to: {file_path}")
        
        data_path = os.path.join(path, "som_node_predicted_cluster_map.npy")
        np.save(data_path, node_predicted_cluster_map)
        print(f"✓ Cluster data saved to: {data_path}")
        
        csv_path = os.path.join(path, "som_node_predicted_cluster_map2.csv")
        np.savetxt(csv_path, node_predicted_cluster_map, delimiter=',', fmt='%d')
        print(f"✓ Cluster data saved as CSV: {csv_path}")
        
        color_path = os.path.join(path, "cluster_colors2.txt")
        with open(color_path, 'w') as f:
            f.write("Cluster,R,G,B\n")
            for cluster, color in cluster_colors.items():
                rgb = [int(c * 255) for c in color[:3]]
                f.write(f"{cluster},{rgb[0]},{rgb[1]},{rgb[2]}\n")
        print(f"✓ Color mapping saved to: {color_path}")
        
    except Exception as e:
        print(f"✗ Failed to save to {path}: {e}")

plt.show(block=False)
plt.pause(1)
plt.close()

# 3.6. cluster statistics
print("\n" + "="*50)
print("CLUSTER STATISTICS:")
print("="*50)
unique_clusters, cluster_counts = np.unique(node_predicted_cluster_map, return_counts=True)
total_nodes = xdim * ydim

print(f"Total SOM nodes: {total_nodes} ({xdim} × {ydim})")
print("\nCluster distribution:")
for cluster, count in zip(unique_clusters, cluster_counts):
    percentage = (count / total_nodes) * 100
    print(f"  Cluster {cluster}: {count} nodes ({percentage:.1f}%)")
    
print(f"\nNumber of clusters found: {len(unique_clusters)}")
print(f"Clusters present: {sorted(unique_clusters.tolist())}")

# Node entropy (Transition Strength)
plt.figure(figsize=(10, 8))
im = plt.imshow(node_entropy, cmap="inferno", origin="lower")
plt.colorbar(im, label="Entropy (Node Mixing)")

plt.title("Node Entropy — Transition / Blended Boundaries")
plt.xlabel("SOM Y Index")
plt.ylabel("SOM X Index")

for path in analysis_dirs:
    plt.savefig(os.path.join(path, "som_node_entropy_map.png"),
                dpi=300, bbox_inches="tight")
plt.show()

cluster_confidence = {}

for c in np.unique(node_predicted_cluster_map):
    mask = node_predicted_cluster_map == c
    total_hits = neuron_counts[mask].sum()
    mean_hits = neuron_counts[mask].mean()
    median_hits = np.median(neuron_counts[mask])
    core_fraction = np.sum(neuron_counts[mask] > 10) / np.sum(mask)  # % of well-supported nodes
    
    cluster_confidence[c] = {
        "total_hits": total_hits,
        "mean_hits_per_node": mean_hits,
        "median_hits": median_hits,
        "core_fraction": core_fraction
    }

print("\nSOM–KMeans Cluster Support (weighted by hit counts)")
print("="*60)
print(f"{'Cluster':<10}{'Total hits':<15}{'Mean/node':<15}{'Median':<10}{'Core frac (>10 hits)':<20}")

for c, stats in cluster_confidence.items():
    print(f"{c:<10}"
          f"{stats['total_hits']:<15}"
          f"{stats['mean_hits_per_node']:<15.2f}"
          f"{stats['median_hits']:<10.1f}"
          f"{stats['core_fraction']*100:<.1f}%")

# ----------------------------------------------------------------------
# Overlay predicted BMU KMeans clusters on U-Matrix
# ----------------------------------------------------------------------
def overlay_kmeans_on_umatrix(som, node_predicted_cluster_map, result_dirs=None):
    """Overlay predicted KMeans clusters on SOM U-Matrix"""

    umatrix, xdim, ydim
   
    plt.figure(figsize=(12, 10))
    
    plt.imshow(umatrix, cmap='bone_r', origin='lower', alpha=0.7)
    plt.colorbar(label="Distance")
    plt.title('SOM U-Matrix with Predicted KMeans Clusters')

    # get dimensions from node_predicted_cluster_map
    xdim_map, ydim_map = node_predicted_cluster_map.shape
    print(f"Node cluster map shape: {node_predicted_cluster_map.shape}")
    print(f"Should match U-Matrix: {umatrix.shape}")
    
    # overlay cluster numbers
    for i in range(xdim):
        for j in range(ydim):
            cluster_num = node_predicted_cluster_map[i, j]
            plt.text(j, i, str(cluster_num), 
                     ha='center', va='center',
                     color='black', fontsize=12
                     )
    
    plt.tight_layout()
    
    if result_dirs:
        for path in analysis_dirs:
            try:
                os.makedirs(path, exist_ok=True)
                file_path = os.path.join(path, "umatrix_predicted_kmeans_overlay.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                print(f"✓ U-Matrix with KMeans overlay saved: {file_path}")
            except Exception as e:
                print(f"✗ Failed to save to {path}: {e}")
    
    plt.show(block=False)
    plt.pause(1)
    plt.close()

overlay_kmeans_on_umatrix(som, node_predicted_cluster_map, result_dirs)

# ----------------------------------------------------------------------
# Env vars gradient corr
# ----------------------------------------------------------------------
def environmental_gradient_analysis(df, cluster_col, method_name, result_dirs=None):
    print(f"\n🎯 ENVIRONMENTAL ANALYSIS: {method_name}")
    print(f"Cluster column: {cluster_col}")
    
    available_env_vars = [var for var in env_vars if var in df.columns]
    print(f"Analyzing {len(available_env_vars)} environmental variables")
    
    if not available_env_vars:
        print("❌ No environmental variables found!")
        return None
    
    # calculate cluster means
    cluster_means = df.groupby(cluster_col)[available_env_vars].mean()
    
    # compute correlations
    corr_results = {}
    for var in available_env_vars:
        corr, p_value = spearmanr(cluster_means.index, cluster_means[var])
        corr_results[var] = {
            "Spearman_r": corr, 
            "p_value": p_value,
            "significance": "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        }
    
    env_corr = pd.DataFrame(corr_results).T
    
    print(f"\n📊 CORRELATION RESULTS:")
    print(env_corr.round(4))
    
    # strong correlations
    strong_corrs = env_corr[(env_corr['p_value'] < 0.05) & (env_corr['Spearman_r'].abs() > 0.5)]
    if not strong_corrs.empty:
        print(f"\n💪 STRONG GRADIENTS (|r| > 0.5):")
        for var, row in strong_corrs.iterrows():
            print(f"  {var}: r = {row['Spearman_r']:.3f}, p = {row['p_value']:.4f}")
    
    plt.figure(figsize=(12, 8))
    
    # correlation plot
    correlations = env_corr['Spearman_r'].sort_values()
    colors = ['red' if x < 0 else 'blue' for x in correlations]
    
    plt.barh(range(len(correlations)), correlations, color=colors, alpha=0.7)
    plt.yticks(range(len(correlations)), correlations.index)
    plt.xlabel('Spearman Correlation')
    plt.title(f'{method_name}\nEnvironmental Gradient Correlations')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # add correlation values
    for i, (var, row) in enumerate(env_corr.loc[correlations.index].iterrows()):
        plt.text(row['Spearman_r'] + (0.02 if row['Spearman_r'] >= 0 else -0.05), 
                i, f"{row['Spearman_r']:.3f} {row['significance']}", 
                ha='left' if row['Spearman_r'] >= 0 else 'right', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if result_dirs:
        for path in analysis_dirs:
            try:
                plt.savefig(os.path.join(path, f"env_gradients_{method_name}.png"), 
                           dpi=300, bbox_inches='tight')
                env_corr.to_csv(os.path.join(path, f"env_correlations_{method_name}.csv"))
                cluster_means.to_csv(os.path.join(path, f"cluster_means_{method_name}.csv"))
                print(f"✅ Saved: {path}")
            except Exception as e:
                print(f"❌ Save error: {e}")
    
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    return env_corr, cluster_means
    
som_env = environmental_gradient_analysis(df2, 'predicted_Cluster', 'manhattan', analysis_dirs)

# ----------------------------------------------------------------------
# Train BMU for KMeans Clustering
# ----------------------------------------------------------------------

def cluster_som_nodes(som, n_clusters=n_clusters):
    """Cluster SOM nodes using K-Means"""
    weights = som.get_weights().reshape(-1, X_hel.shape[1])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(weights)
    return (kmeans.labels_+1).reshape(som._weights.shape[0], som._weights.shape[1])

node_cluster = cluster_som_nodes(som) # cluster label for each node on the SOM from training

# Assign clusters to original data
def assign_clusters(som, data, node_cluster):
    """assign each sample to its cluster"""
    assignments = []
    for x in data:
        winner = som.winner(x)
        assignments.append(node_cluster[winner])
    return assignments

df2['SOM_Cluster'] = assign_clusters(som, X_hel, node_cluster)

# ----------------------------------------------------------------------
# Save final clustered som node dataframe
# ----------------------------------------------------------------------
try:
    final_path = os.path.join(analysis_dirs, "df2_with_clusters.csv")
    df1.to_csv(final_path, index=False)
    print(f"Final DataFrame with SOM clusters saved to: {final_path}")
except Exception as e:
    print(f"Failed to save final DataFrame: {e}")
    
for path in analysis_dirs:
    try:
        df2.to_csv(os.path.join(path, "df2_with_clusters.csv"), index=False)
        print(f"Saved df2 with cluster info to: {path}")
    except Exception as e:
        print(f"Error saving df2_with_clusters: {e}")

# ----------------------------------------------------------------------
# Calculate number of unique clusters and colors
# ----------------------------------------------------------------------
unique_clusters = np.unique(node_cluster)
cluster_ids = np.unique(node_cluster)
n_clusters = len(cluster_ids)

# kmeans cluster color assignment:
tab20 = plt.get_cmap('tab20')
custom_indices = [0, 5, 10, 15, 19, 14, 3]
if n_clusters > len(custom_indices):
    raise ValueError(f"Need {n_clusters} colors but only {len(custom_indices)} provided.")
    
selected_indices = custom_indices[:n_clusters]
discrete_color = [tab20(i) for i in selected_indices]
discrete_tab20 = ListedColormap(discrete_color) # usage: discrete_tab20() i.e. cluster_color(i)/(c1)/(c2)

sorted_clusters = sorted(cluster_ids)
cluster_colors = {
    cluster: discrete_color[i] for i, cluster in enumerate(sorted_clusters)
}

# ----------------------------------------------------------------------
# SOM node clustering (KMeans)
# ----------------------------------------------------------------------
plt.figure(figsize=(12, 12))
plt.imshow(node_cluster,    # create heatmap
           cmap=discrete_tab20, 
           origin='lower',  # Same as U-Matrix
           aspect='auto')

# add annotations
for i in range(xdim):
    for j in range(ydim):
        plt.text(j, i, str(node_cluster[i, j]),
                 ha='center', va='center',
                 color='black' if node_cluster[i, j] % 2 == 0 else 'black')

# ~ plt.colorbar(label='Cluster')
plt.title('Manhattan-based SOM Node Clusters')
# ~ plt.xlabel('SOM X Dimension')
# ~ plt.ylabel('SOM Y Dimension')

# Add grid coordinates for reference (matching U-Matrix)
# ~ plt.xticks(np.arange(node_cluster.shape[1]))
# ~ plt.yticks(np.arange(node_cluster.shape[0]))

for path in analysis_dirs:
    try:
        os.makedirs(path, exist_ok=True)
        
        file_path = os.path.join(path, "som_node_cluster_map.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved successfully to: {file_path}")
        
        data_path = os.path.join(path, "som_node_cluster_map.npy")
        np.save(data_path, node_cluster)
        print(f"✓ Cluster data saved to: {data_path}")
        
        csv_path = os.path.join(path, "som_node_cluster_map.csv")
        np.savetxt(csv_path, node_cluster, delimiter=',', fmt='%d')
        print(f"✓ Cluster data saved as CSV: {csv_path}")
        
        color_path = os.path.join(path, "cluster_colors2.txt")
        with open(color_path, 'w') as f:
            f.write("Cluster,R,G,B\n")
            for cluster, color in cluster_colors.items():
                rgb = [int(c * 255) for c in color[:3]]
                f.write(f"{cluster},{rgb[0]},{rgb[1]},{rgb[2]}\n")
        print(f"✓ Color mapping saved to: {color_path}")
        
    except Exception as e:
        print(f"✗ Failed to save to {path}: {e}")

plt.show(block=False)
plt.pause(1)
plt.close()

# ----------------------------------------------------------------------
# Overlay KMeans clusters on U-Matrix
# ----------------------------------------------------------------------
def overlay_kmeans_on_umatrix(som, node_cluster, result_dirs=None):
    """Overlay KMeans clusters on SOM U-Matrix"""

    umatrix, xdim, ydim
   
    plt.figure(figsize=(12, 10))
    
    plt.imshow(umatrix, cmap='bone_r', origin='lower', alpha=0.7)
    plt.colorbar(label="Distance")
    plt.title('SOM U-Matrix with KMeans Clusters')

    # get dimensions from node_cluster
    xdim_map, ydim_map = node_cluster.shape
    print(f"Node cluster map shape: {node_cluster.shape}")
    print(f"Should match U-Matrix: {umatrix.shape}")
    
    # overlay cluster numbers
    for i in range(xdim):
        for j in range(ydim):
            cluster_num = node_cluster[i, j]
            plt.text(j, i, str(cluster_num), 
                     ha='center', va='center',
                     color='black', fontsize=12
                     )
    
    plt.tight_layout()
    
    if result_dirs:
        for path in analysis_dirs:
            try:
                os.makedirs(path, exist_ok=True)
                file_path = os.path.join(path, "umatrix_kmeans_overlay.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                print(f"✓ U-Matrix with KMeans overlay saved: {file_path}")
            except Exception as e:
                print(f"✗ Failed to save to {path}: {e}")
    
    plt.show(block=False)
    plt.pause(1)
    plt.close()

overlay_kmeans_on_umatrix(som, node_cluster, analysis_dirs)

ari = adjusted_rand_score(df2['cluster'], df2['SOM_Cluster'])
nmi = normalized_mutual_info_score(df2['cluster'], df2['SOM_Cluster'])
print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}")

# ----------------------------------------------------------------------
# SP vars gradient corr (PCA-based, valid gradient)
# ----------------------------------------------------------------------
df4 = pd.read_csv("/home/bernard/Dropbox/IMEE/marine_som/2phase/144000iter/29x27/manhattan_som_sp/nclust4/df2_with_clusters.csv")
def sp_gradient_analysis(
    df,
    gradient_col="PC1",
    method_name="PCA",
    result_dirs=None,
    fdr_correct=True
):
    print(f"\n🎯 ASSEMBLAGE GRADIENT ANALYSIS: {method_name}")
    print(f"Gradient axis: {gradient_col}")

    if gradient_col not in df.columns:
        raise ValueError(f"❌ Gradient column '{gradient_col}' not found in dataframe")

    available_sp_vars = [var for var in sp_vars if var in df.columns]
    print(f"Analyzing {len(available_sp_vars)} assemblage variables")

    if not available_sp_vars:
        print("❌ No assemblage variables found!")
        return None

    gradient = df[gradient_col]

    # compute correlations
    corr_results = []
    for var in available_sp_vars:
        r, p = spearmanr(gradient, df[var], nan_policy="omit")
        if pd.notna(r) and pd.notna(p):
            corr_results.append((var, r, p))

    env_corr = pd.DataFrame(
        corr_results,
        columns=["variable", "Spearman_r", "p_value"]
    ).set_index("variable")

    env_corr["abs_r"] = env_corr["Spearman_r"].abs()

    # FDR correction (recommended for many taxa)
    if fdr_correct:
        from statsmodels.stats.multitest import multipletests
        env_corr["fdr_q"] = multipletests(
            env_corr["p_value"], method="fdr_bh"
        )[1]
        sig_col = "fdr_q"
    else:
        sig_col = "p_value"

    # significance labels
    def sig_label(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    env_corr["significance"] = env_corr[sig_col].apply(sig_label)

    print("\n📊 CORRELATION RESULTS:")
    print(env_corr.round(4))

    # strong gradients
    strong_corrs = env_corr[
        (env_corr[sig_col] < 0.05) &
        (env_corr["Spearman_r"].abs() > 0.5)
    ]

    if not strong_corrs.empty:
        print("\n💪 STRONG GRADIENT DRIVERS (|r| > 0.5):")
        for var, row in strong_corrs.iterrows():
            print(
                f"  {var}: r = {row['Spearman_r']:.3f}, "
                f"p = {row['p_value']:.4g}"
            )

    # ---- FILTER RESULTS FOR PLOTTING ----
    plot_df = env_corr.copy()

    # filter strong & significant correlations
    plot_df = plot_df[
        (plot_df["abs_r"] >= 0.3) &
        (plot_df["fdr_q"] < 0.05)
    ].sort_values("Spearman_r")

    print(f"📉 Plotting {len(plot_df)} strong gradient taxa")

    if plot_df.empty:
        print("❌ No strong correlations to plot")
        return env_corr

    # ---- DYNAMIC FIGURE SIZE ----
    fig_height = max(6, 0.3 * len(plot_df))
    plt.figure(figsize=(10, fig_height))

    # ---- BAR PLOT ----
    colors = ["#d62728" if r < 0 else "#1f77b4" for r in plot_df["Spearman_r"]]

    plt.barh(
        plot_df.index,
        plot_df["Spearman_r"],
        color=colors,
        alpha=0.8
    )

    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel(f"Spearman correlation with {gradient_col}")
    plt.title(
        f"{method_name} – Assemblage associations along {gradient_col}\n"
        f"(|r| ≥ 0.3, FDR q < 0.05)"
    )

    # ---- CLEAN ANNOTATIONS ----
    for y, r in enumerate(plot_df["Spearman_r"]):
        plt.text(
            r + (0.02 if r > 0 else -0.02),
            y,
            f"{r:.2f}",
            va="center",
            ha="left" if r > 0 else "right",
            fontsize=9
        )

    plt.tight_layout()

    # ---- Save outputs ----
    if analysis_dirs:
        for path in analysis_dirs:
            try:
                plt.savefig(
                    os.path.join(path, f"sp_gradients_{method_name}_{gradient_col}.png"),
                    dpi=300,
                    bbox_inches="tight"
                )
                env_corr.to_csv(
                    os.path.join(path, f"env_correlations_{method_name}_{gradient_col}.csv")
                )
                print(f"✅ Saved: {path}")
            except Exception as e:
                print(f"❌ Save error: {e}")

    plt.show(block=False)
    plt.pause(1)
    plt.close()

    return env_corr
som_env = sp_gradient_analysis(
    df4,
    gradient_col="PC1",
    method_name="PCA",
    result_dirs=analysis_dirs
)        
        
#-----------------------------------------------------------------------
# Spatial visualization
# ----------------------------------------------------------------------
# Method 1: Using GeoPandas with natural earth data
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point

try:
    fig, ax = plt.subplots(figsize=(12, 8))

    # create a GeoDataFrame from points
    geometry = [Point(xy) for xy in zip(df2['longitude'], df2['latitude'])]
    gdf = gpd.GeoDataFrame(df2, geometry=geometry, crs="EPSG:4326")

    # plot the points - using your custom colors instead of 'tab20'
    ax = gdf.plot(column='SOM_Cluster', categorical=True, 
                  legend=True, ax=ax,
                  markersize=100, alpha=0.8, 
                  cmap=discrete_tab20, edgecolor='black', linewidth=0.5)  # Use your custom colormap

    # add basemap
    ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)

    plt.title('Spatial Distribution of manhattan-based SOM Clusters - Taiwan')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    for path in analysis_dirs:
        try:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, "spatial_clusters_taiwan_geomap.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Map plot saved successfully to: {file_path}")
        except Exception as e:
            print(f"Failed to save plot to {path}: {e}")
            
    plt.show(block=False)
    plt.pause(2)
    plt.close()

except Exception as e:
    print(f"Error creating geo map: {e}")
    plt.close()  # Ensure cleanup on error

#-----------------------------------------------------------------------
# Method 2: From Taiwan shapefile
#-----------------------------------------------------------------------
def get_dominant_cluster(series):
    """Get the most frequent cluster at each location"""
    mode_result = series.mode()
    return mode_result.iloc[0] if not mode_result.empty else np.nan

# Aggregate to get dominant cluster per location
agg_df = df2.groupby(['longitude', 'latitude']).agg({
    'SOM_Cluster': get_dominant_cluster
}).reset_index()

# Get count of samples per location for sizing
location_counts = df2.groupby(['longitude', 'latitude']).size().reset_index(name='sample_count')
agg_df = agg_df.merge(location_counts, on=['longitude', 'latitude'])

try:
    taiwan_shapefile = "/home/bernard/Documents/thesis/taiwan_boundary"
    taiwan = gpd.read_file(taiwan_shapefile)

    # identify shapefile extent (bounding box)
    minx, miny, maxx, maxy = taiwan.total_bounds

    # manual boundary setting
    minx, maxx = 119.1, 122.4
    miny, maxy = 21.5, 26.0

    plt.figure(figsize=(12, 9))
    ax = plt.gca()

    # plot Taiwan boundary
    taiwan.plot(ax=ax, color='lightgray', edgecolor='gray', alpha=0.6)

    # overlay dominant clusters with size based on sample count
    min_size = 30
    max_size = 150
    sizes = min_size + (agg_df['sample_count'] / agg_df['sample_count'].max()) * (max_size - min_size)
        
    sc = ax.scatter(agg_df['longitude'], agg_df['latitude'], 
                    c=agg_df['SOM_Cluster'], 
                    cmap=discrete_tab20, 
                    s=sizes, alpha=0.8, edgecolor='black', linewidth=0.5)
    # zoom extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    plt.title('Spatial Distribution of SOM-derived Clusters across Taiwan')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    unique_clusters = sorted(agg_df['SOM_Cluster'].dropna().unique())
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=cluster_colors[int(cluster)], 
                                  markersize=8, label=f'Cluster {int(cluster)}') 
                      for cluster in unique_clusters]
    
    # add legend
    ax.legend(handles=legend_elements, title='SOM Clusters', 
              loc='upper left', framealpha=0.9)

    plt.tight_layout()

except Exception as e:
    print(f"Shapefile not available, using simple scatter: {e}")
    plt.figure(figsize=(12, 9))
    
    # Plot without shapefile
    sizes = min_size + (agg_df['sample_count'] / agg_df['sample_count'].max()) * (max_size - min_size)
    
    sc = plt.scatter(agg_df['longitude'], agg_df['latitude'], 
                     c=agg_df['SOM_Cluster'], 
                     cmap=discrete_tab20, 
                     s=sizes, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    plt.title('Spatial Distribution of SOM-derived Clusters across Taiwan')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add legend
    unique_clusters = sorted(agg_df['SOM_Cluster'].dropna().unique())
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=cluster_colors[int(cluster)], 
                                  markersize=8, label=f'Cluster {int(cluster)}') 
                      for cluster in unique_clusters]
    plt.legend(handles=legend_elements, title='SOM Clusters', 
               loc='upper left', framealpha=0.9)
    
for path in analysis_dirs:
    try:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "spatial_clusters_taiwan_shmap.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {file_path}")
    except Exception as e:
        print(f"Failed to save plot to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

# West Taiwan
try:
    taiwan_shapefile = "/home/bernard/Documents/thesis/taiwan_boundary"
    taiwan = gpd.read_file(taiwan_shapefile)

    # identify shapefile extent (bounding box)
    minx, miny, maxx, maxy = taiwan.total_bounds

    # manual boundary setting
    minx, maxx = 119.0, 119.8
    miny, maxy = 23.1, 24.0

    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    # plot Taiwan boundary
    taiwan.plot(ax=ax, color='lightgray', edgecolor='gray', alpha=0.6)

    # overlay dominant clusters with size based on sample count
    min_size = 30
    max_size = 150
    sizes = min_size + (agg_df['sample_count'] / agg_df['sample_count'].max()) * (max_size - min_size)
        
    sc = ax.scatter(agg_df['longitude'], agg_df['latitude'], 
                    c=agg_df['SOM_Cluster'], 
                    cmap=discrete_tab20, 
                    s=sizes, alpha=0.8, edgecolor='black', linewidth=0.5)
    # zoom extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    unique_clusters = sorted(agg_df['SOM_Cluster'].dropna().unique())
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=cluster_colors[int(cluster)], 
                                  markersize=8, label=f'Cluster {int(cluster)}') 
                      for cluster in unique_clusters]
                      
    # add legend
    # ~ ax.legend(handles=legend_elements, title='SOM Clusters', 
              # ~ loc='upper left', framealpha=0.9)

    plt.tight_layout()

except Exception as e:
    print(f"Shapefile not available, using simple scatter: {e}")
    plt.figure(figsize=(5, 5))
    sns.scatterplot(data=df2, x='longitude', y='latitude', hue='SOM_Cluster', 
                    palette=discrete_color, s=50, alpha=0.8)
    plt.title('Spatial Distribution of manhattan-based SOM Clusters')
    
for path in analysis_dirs:
    try:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "spatial_clusters_westtaiwan_shmap.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {file_path}")
    except Exception as e:
        print(f"Failed to save plot to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

#south
try:
    taiwan_shapefile = "/home/bernard/Documents/thesis/taiwan_boundary"
    taiwan = gpd.read_file(taiwan_shapefile)

    # identify shapefile extent (bounding box)
    minx, miny, maxx, maxy = taiwan.total_bounds

    # manual boundary setting
    minx, maxx = 120.1, 121.2
    miny, maxy = 21.7, 22.3

    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    # plot Taiwan boundary
    taiwan.plot(ax=ax, color='lightgray', edgecolor='gray', alpha=0.6)

    # overlay dominant clusters with size based on sample count
    min_size = 30
    max_size = 150
    sizes = min_size + (agg_df['sample_count'] / agg_df['sample_count'].max()) * (max_size - min_size)
        
    sc = ax.scatter(agg_df['longitude'], agg_df['latitude'], 
                    c=agg_df['SOM_Cluster'], 
                    cmap=discrete_tab20, 
                    s=sizes, alpha=0.8, edgecolor='black', linewidth=0.5)
    # zoom extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    unique_clusters = sorted(agg_df['SOM_Cluster'].dropna().unique())
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=cluster_colors[int(cluster)], 
                                  markersize=8, label=f'Cluster {int(cluster)}') 
                      for cluster in unique_clusters]
    
    # add legend
    # ~ ax.legend(handles=legend_elements, title='SOM Clusters', 
              # ~ loc='upper left', framealpha=0.9)

    plt.tight_layout()

except Exception as e:
    print(f"Shapefile not available, using simple scatter: {e}")
    plt.figure(figsize=(12, 9))
    sns.scatterplot(data=df2, x='longitude', y='latitude', hue='SOM_Cluster', 
                    palette=discrete_color, s=50, alpha=0.8)
    plt.title('Spatial Distribution of manhattan-based SOM Clusters')
    
for path in analysis_dirs:
    try:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "spatial_clusters_southtaiwan_shmap.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {file_path}")
    except Exception as e:
        print(f"Failed to save plot to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

#north
try:
    taiwan_shapefile = "/home/bernard/Documents/thesis/taiwan_boundary"
    taiwan = gpd.read_file(taiwan_shapefile)

    # identify shapefile extent (bounding box)
    minx, miny, maxx, maxy = taiwan.total_bounds

    # manual boundary setting
    minx, maxx = 121.4, 122.4
    miny, maxy = 24.8, 25.5

    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    # plot Taiwan boundary
    taiwan.plot(ax=ax, color='lightgray', edgecolor='gray', alpha=0.6)

    # overlay dominant clusters with size based on sample count
    min_size = 30
    max_size = 150
    sizes = min_size + (agg_df['sample_count'] / agg_df['sample_count'].max()) * (max_size - min_size)
        
    sc = ax.scatter(agg_df['longitude'], agg_df['latitude'], 
                    c=agg_df['SOM_Cluster'], 
                    cmap=discrete_tab20, 
                    s=sizes, alpha=0.8, edgecolor='black', linewidth=0.5)
    # zoom extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    unique_clusters = sorted(agg_df['SOM_Cluster'].dropna().unique())
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=cluster_colors[int(cluster)], 
                                  markersize=8, label=f'Cluster {int(cluster)}') 
                      for cluster in unique_clusters]
    
    # add legend
    # ~ ax.legend(handles=legend_elements, title='SOM Clusters', 
              # ~ loc='upper left', framealpha=0.9)

    plt.tight_layout()

except Exception as e:
    print(f"Shapefile not available, using simple scatter: {e}")
    plt.figure(figsize=(12, 9))
    sns.scatterplot(data=df2, x='longitude', y='latitude', hue='SOM_Cluster', 
                    palette=discrete_color, s=50, alpha=0.8)
    plt.title('Spatial Distribution of manhattan-based SOM Clusters')
    
for path in analysis_dirs:
    try:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "spatial_clusters_northtaiwan_shmap.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {file_path}")
    except Exception as e:
        print(f"Failed to save plot to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

# green island
try:
    taiwan_shapefile = "/home/bernard/Documents/thesis/taiwan_boundary"
    taiwan = gpd.read_file(taiwan_shapefile)

    # identify shapefile extent (bounding box)
    minx, miny, maxx, maxy = taiwan.total_bounds

    minx, maxx = 121.44, 121.53   # lon
    miny, maxy = 22.61, 22.70     # lat


    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    # plot Taiwan boundary
    taiwan.plot(ax=ax, color='lightgray', edgecolor='gray', alpha=0.6)

    # overlay dominant clusters with size based on sample count
    min_size = 30
    max_size = 150
    sizes = min_size + (agg_df['sample_count'] / agg_df['sample_count'].max()) * (max_size - min_size)
        
    sc = ax.scatter(agg_df['longitude'], agg_df['latitude'], 
                    c=agg_df['SOM_Cluster'], 
                    cmap=discrete_tab20, 
                    s=sizes, alpha=0.8, edgecolor='black', linewidth=0.5)
    # zoom extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    unique_clusters = sorted(agg_df['SOM_Cluster'].dropna().unique())
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=cluster_colors[int(cluster)], 
                                  markersize=8, label=f'Cluster {int(cluster)}') 
                      for cluster in unique_clusters]
    
    # add legend
    # ~ ax.legend(handles=legend_elements, title='SOM Clusters', 
              # ~ loc='upper left', framealpha=0.9)

    plt.tight_layout()

except Exception as e:
    print(f"Shapefile not available, using simple scatter: {e}")
    plt.figure(figsize=(12, 9))
    sns.scatterplot(data=df2, x='longitude', y='latitude', hue='SOM_Cluster', 
                    palette=discrete_color, s=50, alpha=0.8)
    plt.title('Spatial Distribution of manhattan-based SOM Clusters')
    
for path in analysis_dirs:
    try:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "spatial_clusters_greenislandtaiwan_shmap.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {file_path}")
    except Exception as e:
        print(f"Failed to save plot to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

# orchid island
try:
    taiwan_shapefile = "/home/bernard/Documents/thesis/taiwan_boundary"
    taiwan = gpd.read_file(taiwan_shapefile)

    # identify shapefile extent (bounding box)
    minx, miny, maxx, maxy = taiwan.total_bounds

    minx, maxx = 121.48, 121.63   # lon
    miny, maxy = 21.98, 22.10    # lat

    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    # plot Taiwan boundary
    taiwan.plot(ax=ax, color='lightgray', edgecolor='gray', alpha=0.6)

    # overlay dominant clusters with size based on sample count
    min_size = 30
    max_size = 150
    sizes = min_size + (agg_df['sample_count'] / agg_df['sample_count'].max()) * (max_size - min_size)
        
    sc = ax.scatter(agg_df['longitude'], agg_df['latitude'], 
                    c=agg_df['SOM_Cluster'], 
                    cmap=discrete_tab20, 
                    s=sizes, alpha=0.8, edgecolor='black', linewidth=0.5)
    # zoom extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    unique_clusters = sorted(agg_df['SOM_Cluster'].dropna().unique())
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=cluster_colors[int(cluster)], 
                                  markersize=8, label=f'Cluster {int(cluster)}') 
                      for cluster in unique_clusters]
    
    # add legend
    # ~ ax.legend(handles=legend_elements, title='SOM Clusters', 
              # ~ loc='upper left', framealpha=0.9)

    plt.tight_layout()

except Exception as e:
    print(f"Shapefile not available, using simple scatter: {e}")
    plt.figure(figsize=(12, 9))
    sns.scatterplot(data=df2, x='longitude', y='latitude', hue='SOM_Cluster', 
                    palette=discrete_color, s=50, alpha=0.8)
    plt.title('Spatial Distribution of manhattan-based SOM Clusters')
    
for path in analysis_dirs:
    try:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "spatial_clusters_orchidislandtaiwan_shmap.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {file_path}")
    except Exception as e:
        print(f"Failed to save plot to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

# ----------------------------------------------------------------------
# Species composition heatmap by SOM cluster
# ----------------------------------------------------------------------
df3 = df3[~df3['Label'].isin(nonbio_vars + abiotic)]

df2_cluster = pd.read_csv(os.path.join(analysis_dirs[0], "df2_with_clusters.csv"))
def species_composition_by_cluster(df, feature_cols, label_map, cluster_col):
    """compute average composition per cluster"""
    comp_df = df.groupby(cluster_col)[feature_cols].mean()
    comp_df = comp_df.div(comp_df.sum(axis=1), axis=0)  # relative composition
    
    # map features into major categories
    labels_dict = label_map.set_index('Label')['MajorCategory'].to_dict()
    all_categories = sorted(label_map['MajorCategory'].unique())
    
    # aggregate cluster-level contributions into major categories
    cluster_contributions = []
    for cluster_idx in comp_df.index: # direclty ids number of cluster (indices) 
        contrib = {}
        for feature in feature_cols:
            if feature in labels_dict:
                category = labels_dict[feature]
                contrib[category] = contrib.get(category, 0) + comp_df.loc[cluster_idx, feature]
        cluster_contributions.append(contrib)

    comp_df = pd.DataFrame(cluster_contributions, index=comp_df.index) 
    comp_df = comp_df.fillna(0)
            
    return comp_df

composition_df = species_composition_by_cluster(
    df=df2, 
    feature_cols=sp_vars, 
    label_map=df3,
    cluster_col='SOM_Cluster'
)

plt.figure(figsize=(14, 10))
sns.heatmap(composition_df, cmap='YlGnBu', annot=False, linewidths=0.5)
plt.title('Relative Species Composition by M-based SOM Cluster')
plt.ylabel('SOM Cluster')
plt.xlabel('Major Benthic Categories')
for path in analysis_dirs:
    try:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "composition_by_cluster.png")
        plt.savefig(file_path, dpi=300)
        print(f"Plot saved successfully to: {file_path}")
    except Exception as e:
        print(f"Failed to save plot to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

# ----------------------------------------------------------------------
# Species composition by Major Category per cluster (pie)
# ----------------------------------------------------------------------
df2_cluster = pd.read_csv(os.path.join(analysis_dirs[0], "df2_with_clusters.csv"))
def plot_cluster_major_category_composition(df2, feature_cols, cluster_col, label_map, n_clusters):
    # group mean values per cluster
    cluster_means = df2.groupby(cluster_col)[feature_cols].mean()
    
    # map features into major categories
    labels_dict = df3.set_index('Label')['MajorCategory'].to_dict()
    all_categories = sorted(df3['MajorCategory'].unique())

    # assign colors per category
    cmap = plt.get_cmap("tab20", len(all_categories))
    category_colors = {cat: cmap(i % cmap.N) for i, cat in enumerate(all_categories)}
    
     # aggregate cluster-level contributions into major categories
    cluster_contributions = []
    cluster_ids = cluster_means.index.tolist()
    for cluster_idx in cluster_ids:
        contrib = {}
        for feature in feature_cols:
            category = labels_dict[feature]
            contrib[category] = contrib.get(category, 0) + cluster_means.loc[cluster_idx, feature]

        contrib = {k: v * 100 for k, v in contrib.items()} # convert to percentages
        cluster_contributions.append(contrib)
        
     # plot
    fig, axs = plt.subplots(1, n_clusters, figsize=(6*n_clusters, 10), subplot_kw={'aspect':'equal'})
    if n_clusters == 1:
        axs = [axs]
    
    for i in range(n_clusters):
        contrib = cluster_contributions[i]

        # order categories for consistent colors
        cats, vals = zip(*[(c, contrib[c]) for c in contrib.keys() if contrib[c] > 0.1]) if contrib else ([], []) # only include categories with percentage > 0.1%
        # ~ cats, vals = zip(*[(c, contrib[c]) for c in contrib.keys() if round(contrib[c], 10) > 0]) if contrib else ([], []) # only include categories with percentage > 0%
        colors = [category_colors[c] for c in cats]
        
        wedges, texts = axs[i].pie(vals, colors=colors, startangle=140)
        axs[i].set_title(f'Cluster {i+1} Composition')
        
        # add legend with percentages
        legend_labels = [f'{c}: {contrib[c]:.1f}%' for c in cats]
        axs[i].legend(wedges, legend_labels, loc='upper left', bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    for path in analysis_dirs:
        file_path = os.path.join(path, "composition_by_cluster.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved successfully to: {file_path}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    return cluster_contributions 

cluster_contributions = plot_cluster_major_category_composition(
    df2=df2_cluster,
    label_map=df3,
    feature_cols=sp_vars,
    cluster_col='SOM_Cluster',
    n_clusters=n_clusters
)

# ----------------------------------------------------------------------
# Top 5 major categories only - pie legend
# ----------------------------------------------------------------------
def plot_cluster_major_category_composition(df2, feature_cols, cluster_col, label_map, n_clusters, analysis_dirs):
    # group mean values per cluster
    cluster_means = df2.groupby(cluster_col)[feature_cols].mean()
    
    # map features into major categories
    labels_dict = label_map.set_index('Label')['MajorCategory'].to_dict()
    all_categories = sorted(label_map['MajorCategory'].unique())

    # assign colors per category
    cmap = plt.get_cmap("tab20", len(all_categories))
    category_colors = {cat: cmap(i % cmap.N) for i, cat in enumerate(all_categories)}
    
    # aggregate cluster-level contributions into major categories
    cluster_contributions = []
    cluster_ids = cluster_means.index.tolist()
    for cluster_idx in cluster_ids:
        contrib = {}
        for feature in feature_cols:
            category = labels_dict[feature]
            contrib[category] = contrib.get(category, 0) + cluster_means.loc[cluster_idx, feature]

        contrib = {k: v * 100 for k, v in contrib.items()} # convert to percentages
        cluster_contributions.append(contrib)
        
    # plot
    fig, axs = plt.subplots(1, n_clusters, figsize=(6*n_clusters, 10), subplot_kw={'aspect':'equal'})
    if n_clusters == 1:
        axs = [axs]
    
    for i in range(n_clusters):
        contrib = cluster_contributions[i]

        # get top 5 categories by percentage (excluding zeros)
        sorted_contrib = sorted([(c, p) for c, p in contrib.items() if p > 1e-10], 
                               key=lambda x: x[1], reverse=True)
        top_categories = sorted_contrib[:5]  # Top 5 categories
        
        if top_categories:
            cats, vals = zip(*top_categories)
            colors = [category_colors[c] for c in cats]
            
            # plot
            wedges, texts = axs[i].pie(vals, colors=colors, startangle=140)
            axs[i].set_title(f'Cluster {i+1}', fontweight='bold', fontsize=12)
            
            # add percentage labels on wedges
            for wedge, value in zip(wedges, vals):
                angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
                x = wedge.r * 0.7 * np.cos(np.radians(angle))
                y = wedge.r * 0.7 * np.sin(np.radians(angle))
                axs[i].text(x, y, f'{value:.1f}%', ha='center', va='center', 
                           fontsize=9, fontweight='bold')
            
            # legend with top 5 categories
            legend_labels = [f'{c}: {v:.1f}%' for c, v in top_categories]
            axs[i].legend(wedges, legend_labels, loc='upper left', bbox_to_anchor=(1, 0, 0.5, 1),
                         title="Top Categories", title_fontsize=10)
        else:
            axs[i].set_title(f'Cluster {i+1}')
            axs[i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axs[i].transAxes)
    
    plt.tight_layout()
    for path in analysis_dirs:
        file_path = os.path.join(path, "composition_by_cluster_top5.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved successfully to: {file_path}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    # print top 5 categories for each cluster
    print("\nTop 5 Categories per Cluster:")
    print("=" * 50)
    for i in range(n_clusters):
        contrib = cluster_contributions[i]
        sorted_contrib = sorted([(c, p) for c, p in contrib.items() if p > 1e-10], 
                               key=lambda x: x[1], reverse=True)
        top_categories = sorted_contrib[:5]
        
        print(f"Cluster {i}:")
        for rank, (category, percentage) in enumerate(top_categories, 1):
            print(f"  {rank}. {category}: {percentage:.1f}%")
        print()
    
    return cluster_contributions

cluster_contributions = plot_cluster_major_category_composition(
    df2=df2,
    label_map=df3,
    feature_cols=sp_vars,
    cluster_col='SOM_Cluster',
    n_clusters=n_clusters,
    analysis_dirs=analysis_dirs
)

# ----------------------------------------------------------------------
# Top 5 major cats and "others" pie
# ----------------------------------------------------------------------
def plot_cluster_major_category_composition(df2, feature_cols, cluster_col, label_map, n_clusters, analysis_dirs):
    # group mean values per cluster
    cluster_means = df2.groupby(cluster_col)[feature_cols].mean()
    
    # map features into major categories
    labels_dict = label_map.set_index('Label')['MajorCategory'].to_dict()
    all_categories = sorted(label_map['MajorCategory'].unique())

    # assign colors per category
    cmap = plt.get_cmap("tab20", len(all_categories))  # +1 for "Others"
    category_colors = {cat: cmap(i % cmap.N) for i, cat in enumerate(all_categories)}
    category_colors['Others'] = '#CCCCCC'   # color for Others
    
    # aggregate cluster-level contributions into major categories
    cluster_contributions = []
    cluster_ids = cluster_means.index.tolist()
    for cluster_idx in cluster_ids:
        contrib = {}
        for feature in feature_cols:
            category = labels_dict[feature]
            contrib[category] = contrib.get(category, 0) + cluster_means.loc[cluster_idx, feature]

        contrib = {k: v * 100 for k, v in contrib.items()} # convert to percentages
        cluster_contributions.append(contrib)

    
     # figure size based on grid
    n_rows = 3
    n_cols = 3
    fig_width = 5 * n_cols
    fig_height = 5 * n_rows
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axs = axs.flatten()  # flatten to 1D array
    
    if n_clusters == 1:
        axs = [axs]
    
    for i in range(n_clusters):
        contrib = cluster_contributions[i]

        # get all non-zero categories sorted by percentage
        sorted_contrib = sorted([(c, p) for c, p in contrib.items() if p > 1e-10], 
                               key=lambda x: x[1], reverse=True)
        
        if sorted_contrib:
            # take top 5 and group the rest as "Others"
            top_categories = sorted_contrib[:5]
            other_categories = sorted_contrib[5:]
            
            # calculate "Others" percentage
            others_percentage = sum(v for _, v in other_categories) if other_categories else 0
            
            # prepare data for pie chart
            if others_percentage > 1e-10:
                cats, vals = zip(*top_categories)
                cats = list(cats) + ['Others']
                vals = list(vals) + [others_percentage]
            else:
                cats, vals = zip(*top_categories)
            
            colors = [category_colors.get(c, 'gray') for c in cats]
            
            # plot pie chart
            wedges, texts = axs[i].pie(vals, colors=colors, startangle=140, radius=1.25)
            axs[i].set_title(f'SC {i+1}', fontweight='bold', fontsize=12)
            # ~ axs[i].set_aspect('equal')
            
            # legend
            legend_labels = []
            for j, (category, value) in enumerate(zip(cats, vals)):
                if category == 'Others' and others_percentage > 0:
                    legend_labels.append(f'Others: {value:.1f}%')
                else:
                    legend_labels.append(f'{category}: {value:.1f}%')
            
            axs[i].legend(wedges, legend_labels, loc='upper left', bbox_to_anchor=(1.1, 1),
                         title="Major Categories", title_fontsize=10)
        else:
            axs[i].set_title(f'Cluster {i+1}')
            axs[i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axs[i].transAxes)
            
    # hide any unused subplots
    for j in range(len(cluster_ids), len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    for path in analysis_dirs:
        file_path = os.path.join(path, "composition_by_cluster_top5_others.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved successfully to: {file_path}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    return cluster_contributions

cluster_contributions = plot_cluster_major_category_composition(
    df2=df2,
    label_map=df3,
    feature_cols=sp_vars,
    cluster_col='SOM_Cluster',
    n_clusters=n_clusters,
    analysis_dirs=analysis_dirs
)

# ----------------------------------------------------------------------
# Stacked bar plot - major composition by som cluster
# ----------------------------------------------------------------------
def plot_cluster_major_category_bars(cluster_contributions, label_map):
    contrib_df = pd.DataFrame(cluster_contributions).fillna(0)
    contrib_df.index = [f"Cluster {i}" for i in range(len(cluster_contributions))]

    # assign colors per category
    all_categories = sorted(label_map["MajorCategory"].unique())
    color_map = plt.get_cmap("tab20", len(all_categories))
    category_colors = {cat: color_map(i) for i, cat in enumerate(all_categories)}
    
    # ~ contrib_df.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab20")
    contrib_df = contrib_df.reindex(columns=all_categories, fill_value=0)

    # plot
    contrib_df.plot(
        kind="bar", 
        stacked=True, 
        figsize=(14, 7),
        color=[category_colors[c] for c in contrib_df.columns]
    )

    plt.ylabel("% cover")
    plt.title("Major Category Composition per SOM Cluster")
    plt.legend(
        title="Major Categories", 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left'
    )
    plt.tight_layout()

    for path in analysis_dirs:
        file_path = os.path.join(path, "majorcatcompo_bars.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved successfully to: {file_path}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    return contrib_df

plot_cluster_major_category_bars(
    cluster_contributions,
    label_map=df3
)

def plot_cluster_major_category_bars(cluster_contributions, label_map, analysis_dirs):
    # assign cats with the same colors
    all_categories = sorted(label_map['MajorCategory'].unique())
    
    # assign colors per category - using the same method as pie chart function
    cmap = plt.get_cmap("tab20", len(all_categories))
    category_colors = {cat: cmap(i % cmap.N) for i, cat in enumerate(all_categories)}
    
    # Convert cluster_contributions to DataFrame
    contrib_df = pd.DataFrame(cluster_contributions).fillna(0)
    contrib_df.index = [f"Cluster {i}" for i in range(len(cluster_contributions))]
    
    # Reorder columns to match the order in all_categories for consistent coloring
    # Only include columns that actually have data
    available_categories = [cat for cat in all_categories if cat in contrib_df.columns]
    contrib_df = contrib_df[available_categories]
    
    # Get colors in the same order as the dataframe columns
    colors = [category_colors[cat] for cat in contrib_df.columns]
    
    # plot bar
    fig, ax = plt.subplots(figsize=(12, 6))
    contrib_df.plot(kind="bar", stacked=True, ax=ax, color=colors)
    
    plt.ylabel("Contribution (%)")
    plt.title("Major Category Composition per SOM Cluster (Stacked Bar)")
    plt.legend(title="Major Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    for path in analysis_dirs:
        file_path = os.path.join(path, "majorcatcompo_bars.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Saved stacked bar chart to {file_path}")
    
    plt.show(block=False)
    plt.pause(1)
    plt.close()

plot_cluster_major_category_bars(
    cluster_contributions,
    label_map=df3,
    analysis_dirs=analysis_dirs
)

def plot_cluster_major_category_bars(cluster_contributions, label_map, analysis_dirs):
    # assign categories with the same colors
    all_categories = sorted(label_map['MajorCategory'].unique())
    cmap = plt.get_cmap("tab20", len(all_categories))
    category_colors = {cat: cmap(i % cmap.N) for i, cat in enumerate(all_categories)}
    
    # to df
    contrib_df = pd.DataFrame(cluster_contributions).fillna(0)
    contrib_df.index = [f"Cluster {i}" for i in range(len(cluster_contributions))]
    
    # reorder columns and get colors
    available_categories = [cat for cat in all_categories if cat in contrib_df.columns]
    contrib_df = contrib_df[available_categories]
    colors = [category_colors[cat] for cat in contrib_df.columns]
    
    # plot bar
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = contrib_df.plot(kind="bar", stacked=True, ax=ax, color=colors)
    
    # add percentage labels on bars
    for i, (idx, row) in enumerate(contrib_df.iterrows()):
        cumulative = 0
        for cat in contrib_df.columns:
            value = row[cat]
            if value > 5:  # only label segments larger than 5% for readability
                ax.text(i, cumulative + value/2, f'{value:.1f}%', 
                       ha='center', va='center', fontsize=8, fontweight='bold')
            cumulative += value
    
    plt.ylabel("Contribution (%)")
    plt.title("Major Category Composition per SOM Cluster (Stacked Bar)")
    plt.legend(title="Major Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    for path in analysis_dirs:
        file_path = os.path.join(path, "majorcatcompo_bars.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Saved stacked bar chart to {file_path}")
    
    plt.show(block=False)
    plt.pause(1)
    plt.close()

plot_cluster_major_category_bars(
    cluster_contributions,
    label_map=df3,
    analysis_dirs=analysis_dirs
)

# ----------------------------------------------------------------------
# Overlay sample level KMeans clusters on U-Matrix
# ----------------------------------------------------------------------
def overlay_kmeans_names_on_som(som, df, label_col='cluster'):
    global umatrix
    plt.figure(figsize=(12, 10))
    plt.imshow(umatrix, cmap='bone_r', origin='lower')
    plt.colorbar(label="Distance")
    plt.title('manhattan-based U-Matrix with kmeans labels')

    # BMU assignment
    bmu_labels = defaultdict(list)
    for idx, row in df.iterrows():
        x, y = som.winner(X_hel[idx])
        bmu_labels[(x, y)].append(str(row[label_col])) # all labels added

    # Place labels at the same grid coordinates
    for (x, y), labels in bmu_labels.items():
        label_text = ", ".join(sorted(set(labels))) # removes duplicates
        if len(label_text) > 15:
            label_text = label_text[:12] + "..." # 
        plt.text(y, x, label_text, color='blue', fontsize=8,
                 ha='center', va='center')

    plt.tight_layout()
    for path in analysis_dirs:
        file_path = os.path.join(path, f"u_matrix_overlay_{label_col}.png")
        plt.savefig(file_path, dpi=300)
        print(f"Overlay plot saved: {file_path}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

overlay_kmeans_names_on_som(som, df2, label_col='cluster')

# ----------------------------------------------------------------------
# Overlay SOM nodes colored by dominant KMeans cluster
# ----------------------------------------------------------------------
def overlay_kmeans_clusters_colored(som, df, X_hel, analysis_dirs,
                                    label_col='cluster', n_clusters=5):
    """
    overlay SOM nodes with dominant KMeans cluster color
    """
    global umatrix
    plt.figure(figsize=(12, 10))
    plt.imshow(umatrix, cmap='bone_r', origin='lower')
    plt.colorbar(label="Distance")
    plt.title('manhattan-based U-Matrix by Dominant BMU-level KMeans Cluster')

    # fixed color pallete
    tab20 = plt.get_cmap('tab20')
    custom_indices = [0, 5, 10, 15, 19, 14, 3]
    cluster_colors = [tab20(i) for i in custom_indices[:n_clusters]]
    cmap = ListedColormap(cluster_colors)

    # BMU assignment
    bmu_clusters = defaultdict(list)
    for idx, row in df.iterrows():
        x, y = som.winner(X_hel[idx])
        try:
            bmu_clusters[(x, y)].append(int(row[label_col]))
        except ValueError:
            continue  # skip invalid/missing cluster labels

    # compute dominant cluster per node
    node_dominant_clusters = {}
    for node, clusters in bmu_clusters.items():
        if clusters:         
            counts = np.bincount(clusters, minlength=n_clusters) # bincount - minlength = get count for all clusters
            dominant_cluster = np.argmax(counts) # returns index (cluster number) of max count
            dominance_ratio = counts[dominant_cluster] / np.sum(counts)
            node_dominant_clusters[node] = {
                'dominant_cluster': dominant_cluster,
                'dominance_ratio': dominance_ratio,
                'sample_count': len(clusters)
            }

    # plot
    for (x, y), info in node_dominant_clusters.items():
        color = cluster_colors[info['dominant_cluster'] % len(cluster_colors)]
        alpha = 0.5 + 0.5 * info['dominance_ratio']  # more opaque = more dominant
        plt.scatter(y, x, s=500*2, color=color,
                    edgecolors='black', linewidths=0.8, alpha=alpha)

        # label number of cluster samples
        plt.text(y, x, f"n={info['sample_count']}",
                 ha='center', va='center', fontsize=7, color='black')

    # legend
    handles = [
        mpatches.Patch(color=cluster_colors[i], label=f'Cluster {i}')
        for i in range(n_clusters)
    ]
    plt.legend(handles=handles, title="KMeans Clusters",
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    for path in analysis_dirs:
        file_path = os.path.join(path, "u_matrix_kmeans_colored.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Colored cluster overlay saved: {file_path}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

overlay_kmeans_clusters_colored(
    som=som,
    df=df2,
    X_hel=X_hel,
    analysis_dirs=analysis_dirs,
    label_col='cluster',
    n_clusters=n_clusters
)

# ----------------------------------------------------------------------
# U-Matrix + selected driver overlays
# ----------------------------------------------------------------------
'''
def plot_category_overlays(som, df, category_list, X_hel):
    fig, axes = plt.subplots(1, len(category_list) + 1, figsize=(5 * (len(category_list) + 1), 5))

    # Panel A: U-Matrix
    axes[0].pcolor(som.distance_map().T, cmap='bone_r')
    axes[0].set_title('U-Matrix')

    for i, category in enumerate(category_list):
        category_density = np.zeros((som._weights.shape[0], som._weights.shape[1]))

        for idx, row in df.iterrows():
            x, y = som.winner(X_hel[idx])
            category_density[x, y] += row.get(category, 0)

        axes[i + 1].imshowshow(category_density.T, origin='lower', cmap='viridis')
        axes[i + 1].set_title(category)

    plt.tight_layout()
    for path in analysis_dirs:
        try:
            file_path = os.path.join(path, "category_overlay_panels.png")
            plt.savefig(file_path, dpi=300)
            print(f"Category overlay plot saved: {file_path}")
        except Exception as e:
            print(f"Error saving category overlay plot: {e}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()
'''
'''
def plot_category_overlays(som, df, category_list, X_hel, n_cols=4):
    n_categories = len(category_list)
    n_panels = n_categories + 1  # +1 for U-Matrix
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    # Panel A: U-Matrix
    axes[0].pcolor(som.distance_map().T, cmap='bone_r')
    axes[0].set_title('U-Matrix')

    for i, category in enumerate(category_list):
        category_density = np.zeros((som._weights.shape[0], som._weights.shape[1]))

        for idx, row in df.iterrows():
            x, y = som.winner(X_hel[idx])
            category_density[x, y] += row.get(category, 0)

        axes[i + 1].imshow(category_density.T, origin='lower', cmap='viridis')
        axes[i + 1].set_title(category)

    # turn off empty axes
    for j in range(n_panels, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    for path in analysis_dirs:
        try:
            file_path = os.path.join(path, "category_overlay_panels.png")
            plt.savefig(file_path, dpi=300)
            print(f"Category overlay plot saved: {file_path}")
        except Exception as e:
            print(f"Error saving category overlay plot: {e}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()
'''

def plot_category_overlays(som, df, category_list, X_hel, n_cols=4):
    n_categories = len(category_list)
    n_panels = n_categories + 1  # +1 for U-Matrix
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Panel A: U-Matrix
    u_matrix = som.distance_map()
    im0 = axes[0].imshow(u_matrix, cmap='bone_r', origin='lower')
    axes[0].set_title('manhattan-based U-Matrix', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    for i, category in enumerate(category_list):
        category_density = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1]))

        for idx, row in df.iterrows():
            x, y = som.winner(X_hel[idx])
            category_density[x, y] += row.get(category, 0)

        im = axes[i + 1].imshow(category_density, origin='lower', cmap='viridis')
        axes[i + 1].set_title(category, fontsize=12)
        axes[i + 1].axis('off')
        fig.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)

    # turn off empty axes
    for j in range(n_panels, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    for path in analysis_dirs:
        try:
            file_path = os.path.join(path, "driver_overlay_panels_cleaned.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Category overlay plot saved: {file_path}")
        except Exception as e:
            print(f"Error saving category overlay plot: {e}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

plot_category_overlays(som, df2, env_vars, X_hel, n_cols=4)


def analyze_cluster_category_composition(df, cluster_col, species_vars, labels_dict, result_dirs=None):
    """
    Analyze cluster composition by major categories
    """
    print(f"\n📊 CLUSTER COMPOSITION BY MAJOR CATEGORIES")
    
    # Map species to categories
    category_contributions = {}
    for species in species_vars:
        if species in labels_dict:
            category = labels_dict[species]
            if category not in category_contributions:
                category_contributions[category] = []
            category_contributions[category].append(species)
    
    # Calculate cluster means by category
    cluster_means_by_category = {}
    unique_clusters = sorted(df[cluster_col].unique())
    
    for cluster in unique_clusters:
        cluster_data = df[df[cluster_col] == cluster]
        category_means = {}
        
        for category, species_list in category_contributions.items():
            available_species = [sp for sp in species_list if sp in cluster_data.columns]
            if available_species:
                category_means[category] = cluster_data[available_species].mean().mean()
        
        cluster_means_by_category[cluster] = category_means
    
    # Create visualization
    category_df = pd.DataFrame(cluster_means_by_category).T
    category_df = category_df.fillna(0)
    
    # Convert to percentages
    category_pct = category_df.div(category_df.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(12, 8))
    category_pct.plot(kind='bar', stacked=True, ax=plt.gca(), 
                     colormap='tab20', alpha=0.8)
    plt.title('Cluster Composition by Major Categories', fontweight='bold', pad=20)
    plt.xlabel('Cluster ID')
    plt.ylabel('Percentage Composition (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save results
    if result_dirs:
        for path in analysis_dirs:
            try:
                plt.savefig(os.path.join(path, "cluster_category_composition.png"), 
                           dpi=300, bbox_inches='tight')
                category_pct.to_csv(os.path.join(path, "cluster_category_percentages.csv"))
                print(f"✓ Cluster composition saved: {path}")
            except Exception as e:
                print(f"✗ Error saving to {path}: {e}")
    
    plt.show()
    
    return category_pct


# 1. Detailed species-level analysis
# ~ detailed_correlations = plot_species_environment_correlation(
    # ~ bmu_sp_avg, bmu_env_avg, sp_vars, env_vars, 
    # ~ labels_dict=None, threshold=0.5
# ~ )

# 2. Aggregated by major categories (using your labels_dict)
# First create labels_dict from your data
label_map=df3
label_map = label_map[~label_map['Label'].isin(nonbio_vars + abiotic)]
labels_dict = label_map.set_index('Label')['MajorCategory'].to_dict()

# ~ aggregated_correlations = plot_species_environment_correlation(
    # ~ bmu_species_avg, bmu_env_avg, sp_vars, env_vars,
    # ~ labels_dict=labels_dict, threshold=0.5
# ~ )

# 3. Cluster composition analysis
cluster_composition = analyze_cluster_category_composition(
    df2, 'SOM_Cluster', sp_vars, labels_dict, analysis_dirs
)

def identify_signature_species_per_cluster(df, cluster_col, species_vars, labels_dict, result_dirs=None, top_n=5):
    """
    Identify signature/representative species or cover types per cluster
    """
    print(f"\n🔍 IDENTIFYING SIGNATURE SPECIES PER CLUSTER")
    
    unique_clusters = sorted(df[cluster_col].unique())
    all_results = {}
    
    for cluster in unique_clusters:
        print(f"\n📈 Analyzing Cluster {cluster}:")
        
        cluster_data = df[df[cluster_col] == cluster]
        global_means = df[species_vars].mean()
        
        # Calculate cluster means and enrichment scores
        cluster_means = cluster_data[species_vars].mean()
        enrichment_scores = (cluster_means / global_means).fillna(0)
        
        # Calculate contribution to total abundance
        total_abundance = cluster_means.sum()
        percentage_contributions = (cluster_means / total_abundance * 100).fillna(0)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'species': species_vars,
            'cluster_mean': cluster_means.values,
            'global_mean': global_means.values,
            'enrichment_score': enrichment_scores.values,
            'percentage_contribution': percentage_contributions.values
        })
        
        # Add category information
        results['category'] = results['species'].map(labels_dict)
        
        # Sort by different criteria to find signature species
        # 1. Most enriched (highest enrichment score)
        top_enriched = results.nlargest(top_n, 'enrichment_score')
        
        # 2. Highest absolute abundance in cluster
        top_abundant = results.nlargest(top_n, 'cluster_mean')
        
        # 3. Highest percentage contribution
        top_contributors = results.nlargest(top_n, 'percentage_contribution')
        
        # Store results
        all_results[cluster] = {
            'top_enriched': top_enriched,
            'top_abundant': top_abundant,
            'top_contributors': top_contributors,
            'full_results': results
        }
        
        # Print summary
        print(f"   Size: {len(cluster_data)} samples")
        print(f"   Total abundance: {total_abundance:.2f}")
        
        print(f"   🎯 Most enriched species (enrichment score):")
        for _, row in top_enriched.iterrows():
            print(f"      {row['species']:20} ({row['category']:25}): {row['enrichment_score']:.2f}x")
        
        print(f"   📊 Most abundant species (absolute mean):")
        for _, row in top_abundant.iterrows():
            print(f"      {row['species']:20} ({row['category']:25}): {row['cluster_mean']:.3f}")
    
    # Create visualizations
    create_signature_species_visualizations(all_results, result_dirs)
    
    return all_results

'''
def create_signature_species_visualizations(all_results, result_dirs):
    """Create visualizations for signature species analysis"""
    
    # 1. Top enriched species per cluster
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Plot 1: Enrichment scores heatmap
    enrichment_data = []
    for cluster, results in all_results.items():
        top_enriched = results['top_enriched']
        for _, row in top_enriched.iterrows():
            enrichment_data.append({
                'cluster': cluster,
                'species': row['species'],
                'category': row['category'],
                'enrichment_score': row['enrichment_score']
            })
    
    if enrichment_data:
        enrichment_df = pd.DataFrame(enrichment_data)
        pivot_enrich = enrichment_df.pivot(index='species', columns='cluster', values='enrichment_score')
        
        sns.heatmap(pivot_enrich.fillna(0), ax=axes[0,0], cmap='YlOrRd', annot=True, fmt='.1f')
        axes[0,0].set_title('Top Enriched Species per Cluster\n(Enrichment Score)', fontweight='bold')
        axes[0,0].set_ylabel('Species')
    
    # Plot 2: Category composition of top species
    category_counts = []
    for cluster, results in all_results.items():
        top_enriched = results['top_enriched']
        category_dist = top_enriched['category'].value_counts()
        for category, count in category_dist.items():
            category_counts.append({
                'cluster': cluster,
                'category': category,
                'count': count
            })
    
    if category_counts:
        category_df = pd.DataFrame(category_counts)
        pivot_cat = category_df.pivot(index='category', columns='cluster', values='count').fillna(0)
        sns.heatmap(pivot_cat, ax=axes[0,1], cmap='Blues', annot=True, fmt='.0f')
        axes[0,1].set_title('Category Distribution of Top Enriched Species', fontweight='bold')
    
    # Plot 3: Enrichment score distribution by category
    category_enrichment = []
    for cluster, results in all_results.items():
        full_results = results['full_results']
        for category in full_results['category'].unique():
            cat_data = full_results[full_results['category'] == category]
            mean_enrichment = cat_data['enrichment_score'].mean()
            category_enrichment.append({
                'cluster': cluster,
                'category': category,
                'mean_enrichment': mean_enrichment
            })
    
    if category_enrichment:
        cat_enrich_df = pd.DataFrame(category_enrichment)
        pivot_cat_enrich = cat_enrich_df.pivot(index='category', columns='cluster', values='mean_enrichment')
        sns.heatmap(pivot_cat_enrich.fillna(0), ax=axes[1,0], cmap='RdYlBu', center=1, annot=True, fmt='.2f')
        axes[1,0].set_title('Mean Enrichment Score by Category', fontweight='bold')
        axes[1,0].set_ylabel('Category')
    
    # Plot 4: Cluster characteristics summary
    cluster_stats = []
    for cluster, results in all_results.items():
        full_results = results['full_results']
        cluster_stats.append({
            'cluster': cluster,
            'total_abundance': full_results['cluster_mean'].sum(),
            'mean_enrichment': full_results['enrichment_score'].mean(),
            'n_highly_enriched': len(full_results[full_results['enrichment_score'] > 2]),
            'dominant_category': full_results.groupby('category')['percentage_contribution'].sum().idxmax()
        })
    
    if cluster_stats:
        stats_df = pd.DataFrame(cluster_stats)
        axes[1,1].axis('off')
        table = axes[1,1].table(cellText=stats_df.values,
                               colLabels=stats_df.columns,
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        axes[1,1].set_title('Cluster Characteristics Summary', fontweight='bold')
    
    plt.tight_layout()
    
    # Save results
    if result_dirs:
        for path in result_dirs:
            try:
                plt.savefig(os.path.join(path, "signature_species_analysis.png"), 
                           dpi=300, bbox_inches='tight')
               
                for cluster, results in all_results.items():
                    results['full_results'].to_csv(
                        os.path.join(path, f"cluster_{cluster}_species_analysis.csv"), 
                        index=False
                    )
                
                summary_data = []
                for cluster, results in all_results.items():
                    top_enriched = results['top_enriched']
                    for _, row in top_enriched.iterrows():
                        summary_data.append({
                            'cluster': cluster,
                            'species': row['species'],
                            'category': row['category'],
                            'enrichment_score': row['enrichment_score'],
                            'cluster_mean': row['cluster_mean'],
                            'percentage_contribution': row['percentage_contribution']
                        })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(os.path.join(path, "signature_species_summary.csv"), index=False)
                
                print(f"✓ Signature species analysis saved: {path}")
                
            except Exception as e:
                print(f"✗ Error saving to {path}: {e}")
    
    plt.show()
    
    return all_results
'''
def create_signature_species_visualizations(all_results, result_dirs):
    """separate visualizations for signature species analysis"""

    # -----------------------------
    # Plot 1: Enrichment scores heatmap
    # -----------------------------
    enrichment_data = []
    for cluster, results in all_results.items():
        top_enriched = results['top_enriched']
        for _, row in top_enriched.iterrows():
            enrichment_data.append({
                'cluster': cluster,
                'species': row['species'],
                'category': row['category'],
                'enrichment_score': row['enrichment_score']
            })

    if enrichment_data:
        enrichment_df = pd.DataFrame(enrichment_data)
        pivot_enrich = enrichment_df.pivot(
            index='species', columns='cluster', values='enrichment_score'
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_enrich.fillna(0), cmap='YlOrRd', annot=True, fmt='.1f')
        plt.title('Top Enriched Operational Taxonomic Units (OTUs) per SOM-derived Community', fontweight='bold')
        plt.ylabel('OTU labels')
        plt.xlabel('SOM-derived Cluster')
        plt.tight_layout()

        if result_dirs:
            for path in result_dirs:
                plt.savefig(
                    os.path.join(path, "signature_species_enrichment_heatmap.png"),
                    dpi=300, bbox_inches='tight'
                )
        plt.show()

    # -----------------------------
    # Plot 2: Category composition of top species
    # -----------------------------
    category_counts = []
    for cluster, results in all_results.items():
        top_enriched = results['top_enriched']
        category_dist = top_enriched['category'].value_counts()
        for category, count in category_dist.items():
            category_counts.append({
                'cluster': cluster,
                'category': category,
                'count': count
            })

    if category_counts:
        category_df = pd.DataFrame(category_counts)
        pivot_cat = category_df.pivot(
            index='category', columns='cluster', values='count'
        ).fillna(0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_cat, cmap='Blues', annot=True, fmt='.0f')
        plt.title('Distribution of Top Enriched Major Categories', fontweight='bold')
        plt.ylabel('Major Category')
        plt.xlabel('Cluster')
        plt.tight_layout()

        if result_dirs:
            for path in result_dirs:
                plt.savefig(
                    os.path.join(path, "signature_species_category_distribution.png"),
                    dpi=300, bbox_inches='tight'
                )
        plt.show()

    # -----------------------------
    # Plot 3: Mean enrichment score by category
    # -----------------------------
    category_enrichment = []
    for cluster, results in all_results.items():
        full_results = results['full_results']
        for category in full_results['category'].unique():
            cat_data = full_results[full_results['category'] == category]
            category_enrichment.append({
                'cluster': cluster,
                'category': category,
                'mean_enrichment': cat_data['enrichment_score'].mean()
            })

    if category_enrichment:
        cat_enrich_df = pd.DataFrame(category_enrichment)
        pivot_cat_enrich = cat_enrich_df.pivot(
            index='category', columns='cluster', values='mean_enrichment'
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_cat_enrich.fillna(0),
            cmap='RdYlBu',
            center=1,
            annot=True,
            fmt='.2f'
        )
        plt.title('Mean Enrichment Score by Major Category', fontweight='bold')
        plt.ylabel('Major Category')
        plt.xlabel('Cluster')
        plt.tight_layout()

        if result_dirs:
            for path in result_dirs:
                plt.savefig(
                    os.path.join(path, "signature_species_mean_enrichment_by_category.png"),
                    dpi=300, bbox_inches='tight'
                )
        plt.show()

    # -----------------------------
    # Plot 4: Cluster characteristics summary table
    # -----------------------------
    cluster_stats = []
    for cluster, results in all_results.items():
        full_results = results['full_results']
        cluster_stats.append({
            'cluster': cluster,
            'total_abundance': full_results['cluster_mean'].sum(),
            'mean_enrichment': full_results['enrichment_score'].mean(),
            'n_highly_enriched': len(full_results[full_results['enrichment_score'] > 2]),
            'dominant_category': (
                full_results.groupby('category')['percentage_contribution']
                .sum()
                .idxmax()
            )
        })

    if cluster_stats:
        stats_df = pd.DataFrame(cluster_stats)

        plt.figure(figsize=(10, 4))
        plt.axis('off')
        table = plt.table(
            cellText=stats_df.values,
            colLabels=stats_df.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        plt.title('Cluster Characteristics Summary', fontweight='bold')
        plt.tight_layout()

        if result_dirs:
            for path in result_dirs:
                plt.savefig(
                    os.path.join(path, "signature_species_cluster_summary_table.png"),
                    dpi=300, bbox_inches='tight'
                )
        plt.show()

    # -----------------------------
    # Save CSV outputs (unchanged)
    # -----------------------------
    if result_dirs:
        for path in result_dirs:
            for cluster, results in all_results.items():
                results['full_results'].to_csv(
                    os.path.join(path, f"cluster_{cluster}_species_analysis.csv"),
                    index=False
                )

            summary_data = []
            for cluster, results in all_results.items():
                for _, row in results['top_enriched'].iterrows():
                    summary_data.append({
                        'cluster': cluster,
                        'species': row['species'],
                        'category': row['category'],
                        'enrichment_score': row['enrichment_score'],
                        'cluster_mean': row['cluster_mean'],
                        'percentage_contribution': row['percentage_contribution']
                    })

            pd.DataFrame(summary_data).to_csv(
                os.path.join(path, "signature_species_summary.csv"),
                index=False
            )

            print(f"✓ Signature species analysis saved: {path}")

    return all_results

signature_results = identify_signature_species_per_cluster(
    df=df2,
    cluster_col='SOM_Cluster',  # or 'cluster' for K-means
    species_vars=sp_vars,       # species variables
    labels_dict=labels_dict,    # category mapping
    result_dirs=analysis_dirs,
    top_n=5
)

# ======================================================================
# Transition analysis
# ======================================================================

# ----------------------------------------------------------------------
# Calculate cluster centers and vectors
# ----------------------------------------------------------------------

labels = df2['SOM_Cluster'].values
unique_labels = np.unique(labels)
n_clusters = len(unique_labels)
label_to_index = {label: i for i, label in enumerate(unique_labels)}
sp_vars = [v for v in sp_vars if v in df3['Label'].values]

# calculate cluster centers in SOM grid space:
'''draw arrows between clusters or place labels'''
cluster_centers = {}
for cluster_id in range(n_clusters):
    coords = np.argwhere(node_cluster == cluster_id)
    if len(coords) > 0:
        centroid = coords.mean(axis=0)
        medoid = coords[np.argmin(np.linalg.norm(coords - centroid, axis=1))]

# calculate cluster vectors in feature space:
'''averages SOM weight vectors of each node in the cluster: to compare clusters'''
cluster_vectors = {}
for cluster_id in range(n_clusters):
    indices = np.argwhere(node_cluster == cluster_id).reshape(-1, 2)
    if len(indices) > 0:
        node_vectors = [som.get_weights()[x, y] for x, y in indices]
        cluster_vectors[cluster_id] = np.mean(node_vectors, axis=0)

# ----------------------------------------------------------------------
# Transition zone analysis
# ----------------------------------------------------------------------
def find_transition_pairs(node_cluster):
    """detect all unique adjacent cluster pairs in the SOM"""
    transitions = set()
    for i in range(node_cluster.shape[0]):
        for j in range(node_cluster.shape[1]):
            current = node_cluster[i, j]
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:  # check 4-neighborhood
                ni, nj = i + dx, j + dy
                if 0 <= ni < node_cluster.shape[0] and 0 <= nj < node_cluster.shape[1]:
                    neighbor = node_cluster[ni, nj]
                    if neighbor != current:
                        pair = tuple(sorted((current, neighbor)))
                        transitions.add(pair)
    return sorted(transitions)

def get_neighboring_clusters(node, node_cluster):
    """get set of neighboring cluster IDs for a node: detects transition zones """
    x, y = node
    neighbors = set()
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < node_cluster.shape[0] and 0 <= ny < node_cluster.shape[1]:
                neighbors.add(node_cluster[nx, ny])
    return neighbors

# ~ '''    

# Identify transition(edge) components between two clusters only
def plot_transition_zone(cluster_A, cluster_B, node_cluster, bmu_locations, 
                        umatrix, xx, yy, xrange, yrange, X_hel, df2, 
                        variables_of_interest, som_shape, figsize=(14, 12)):
    """transition zone visualization consistent with kmeans (tab20) cluster coloring"""

    # kmeans cluster color assignment:
    labels = df2['SOM_Cluster'].values
    unique_labels = np.unique(labels)
    cluster_ids = np.unique(node_cluster)
    n_clusters = len(unique_labels)
    tab20 = plt.get_cmap('tab20')
    custom_indices = [0, 5, 10, 15, 19, 14, 3, 7, 9, 11, 13, 17, 2, 4]
    selected_indices = custom_indices[:7]
    if n_clusters > len(custom_indices):
        raise ValueError(f"Need {n_clusters} colors but only {len(custom_indices)} provided.")

    discrete_color = [tab20(i) for i in selected_indices]
    discrete_tab20 = ListedColormap(discrete_color) # usage: discrete_tab20() i.e. cluster_color(i)/(c1)/(c2)

    sorted_clusters = sorted(unique_clusters)
    cluster_colors = {
        cluster: discrete_color[i] for i, cluster in enumerate(sorted_clusters)
    }
    cluster_alphas = {cluster: 0.25 if cluster in [cluster_A, cluster_B] else 0.1 for cluster in cluster_ids}

    n_clusters = len(cluster_ids)
    if n_clusters > len(custom_indices):
        raise ValueError(
            f"Need {n_clusters} colors but only {len(custom_indices)} provided."
        )

    # select exactly as many colors as clusters
    selected_indices = custom_indices[:n_clusters]
    discrete_color = [tab20(i) for i in selected_indices]
    discrete_tab20 = ListedColormap(discrete_color)

    # stable cluster → color mapping
    sorted_clusters = sorted(cluster_ids)
    cluster_colors = {
        cluster: discrete_color[i]
        for i, cluster in enumerate(sorted_clusters)
    }

    # transparency control
    cluster_alphas = {
        cluster: 0.25 if cluster in [cluster_A, cluster_B] else 0.1
        for cluster in cluster_ids
    }

    # 1. identify boundaries
    transitional_samples = []
    transitional_weights = []
    transition_nodes = set()
    
    for idx, (x, y) in enumerate(bmu_locations):
        neighbors = get_neighboring_clusters((x, y), node_cluster)
        if cluster_A in neighbors and cluster_B in neighbors:
            transitional_samples.append(idx)
            transitional_weights.append(X_hel[idx])
            transition_nodes.add((x, y))
    
    if not transitional_samples:
        print(f"No transition samples between Cluster {cluster_A} and {cluster_B}")
        return None
    
    # 2. create figure with adjusted size for better font scaling
    plt.switch_backend('TkAgg')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
    
    # Calculate font sizes based on figure dimensions
    fig_width, fig_height = fig.get_size_inches()
    base_font_size = max(12, min(16, fig_width * 1.2))  # Dynamic scaling
    title_font_size = base_font_size + 2
    label_font_size = base_font_size
    tick_font_size = base_font_size - 2
    legend_font_size = base_font_size - 1
    
    # 3. left subplot: SOM visualization
    norm_umatrix = (umatrix - umatrix.min()) / (umatrix.max() - umatrix.min())
    
    # plot rectangular U-Matrix background
    ax1.imshow(umatrix, cmap='bone_r', alpha=0.7,
              extent=[xrange[0]-0.5, xrange[-1]+0.5, 
                     yrange[-1]+0.5, yrange[0]-0.5],
              aspect='auto',
              origin='lower')
    
    # highlight all clusters
    for cluster in cluster_ids:
        color = cluster_colors[cluster]
        alpha = cluster_alphas[cluster]
        for (x, y) in np.argwhere(node_cluster == cluster):
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                facecolor=color,
                                alpha=alpha,
                                edgecolor='none')
            ax1.add_patch(rect)
    
    # highlight focus clusters more prominently
    for cluster in [cluster_A, cluster_B]:
        color = cluster_colors[cluster]
        for (x, y) in np.argwhere(node_cluster == cluster):
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                facecolor=color, alpha=0.6,
                                edgecolor='black', linewidth=1)
            ax1.add_patch(rect)
            
    # highlight transition nodes
    for cluster in [cluster_A, cluster_B]:
        color = cluster_colors[cluster]
        for (x, y) in np.argwhere(node_cluster == cluster):
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                facecolor=color, alpha=0.6,
                                edgecolor='black', linewidth=1)
            ax1.add_patch(rect)
    
    # legend
    legend_elements = [
        Patch(facecolor=cluster_colors[cluster_A], alpha=0.6, label=f'SC {cluster_A}'),
        Patch(facecolor=cluster_colors[cluster_B], alpha=0.6, label=f'SC {cluster_B}'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9, 
               fontsize=legend_font_size, prop={'weight': 'bold'})
    
    ax1.set_title(f'Transition Zone: SC {cluster_A} ↔ {cluster_B}\n'
                f'{len(transitional_samples)} Transition Samples', 
                pad=15, fontsize=title_font_size, fontweight='bold')
    ax1.set_xlabel('X Coordinate', fontsize=label_font_size, fontweight='bold')
    ax1.set_ylabel('Y Coordinate', fontsize=label_font_size, fontweight='bold')
    ax1.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
    
    # Set tick label font size
    ax1.tick_params(axis='both', which='major', labelsize=tick_font_size)
    
    # --- right subplot: Feature Differences
    transition_df = df2.iloc[transitional_samples] # extract feature values
    feature_means = transition_df[variables_of_interest].mean().sort_values(ascending=False)
    global_means = df2[variables_of_interest].mean()
    feature_diff = (
        transition_df[variables_of_interest].mean()
        - df2[variables_of_interest].mean()
    ) / df2[variables_of_interest].std()

    top_n = 5
    top_features = feature_diff.sort_values(ascending=False).head(top_n)

    max_val = np.max(top_features.values)
    min_val = np.min(top_features.values)

    cmap = plt.get_cmap('viridis')

    colors = []
    for val in top_features.values:
        if max_val > min_val:
            # Simple linear scaling
            intensity = (val - min_val) / (max_val - min_val)
        else:
            intensity = 0.5
        # Skip the very dark end of the colormap
        colors.append(cmap(0.2 + 0.8 * intensity))
    
    # Create bars with increased height for better readability
    bar_height = 0.7
    bars = ax2.barh(top_features.index, top_features.values, 
                    color=colors, edgecolor='black', height=bar_height)

    # Add value labels with dynamic font sizing
    for bar in bars:
        width = bar.get_width()
        # Calculate font size based on bar width and figure size
        bar_font_size = max(base_font_size - 1, min(base_font_size + 2, 
                                                   base_font_size * (1 + abs(width)/max_val * 0.5)))
        
        # Choose label position based on bar width
        if width > max_val * 0.3:  # Large bars
            label_x = width * 0.4
            text_color = 'white'
        else:  # Small bars
            label_x = width * 0.5
            text_color = 'black'
            
        ax2.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}σ', 
                ha='center', va='center', 
                color=text_color, 
                fontweight='bold', 
                fontsize=bar_font_size)

    ax2.set_xlabel('Relative Difference from Global Mean (σ)', 
                  fontsize=label_font_size + 1, fontweight='bold')
    ax2.set_title(f'Top {top_n} Discriminating Features\nSC {cluster_A} ↔ SC {cluster_B}', 
                 fontsize=title_font_size, fontweight='bold')
    ax2.axvline(0, color='gray', linestyle='--', linewidth=1.5)
    
    # Increase y-axis tick label font size for better readability
    ax2.tick_params(axis='y', which='major', labelsize=tick_font_size + 1)
    ax2.tick_params(axis='x', which='major', labelsize=tick_font_size)
    
    # Make x-axis label bold
    ax2.xaxis.label.set_fontweight('bold')
    
    # Adjust layout with more padding for larger fonts
    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=3.0)
    
    for path in analysis_dirs:
        try:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, f"transition_{cluster_A}_to_{cluster_B}.png")
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved successfully to: {file_path}")
        except Exception as e:
            print(f"Failed to save plot to {path}: {e}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    print(f"Saved: transition_{cluster_A}_to_{cluster_B}.png")

    # save vectors
    transition_vectors = np.array(transitional_weights)
    transition_df = pd.DataFrame(transition_vectors, columns=variables_of_interest)

    for path in analysis_dirs:
        try:
            os.makedirs(path, exist_ok=True)
            vector_file = os.path.join(path, f"transition_vectors_cluster_{cluster_A}_to_{cluster_B}.csv")
            transition_df.to_csv(vector_file, index=False)
            print(f"Transition vectors saved to: {vector_file}")
        except Exception as e:
            print(f"Failed to save transition vectors to {path}: {e}")

# ~ '''
# Plot transition zones for different cluster pairs
plot_transition_zone(cluster_A=1, cluster_B=2, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel,
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=1, cluster_B=3, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=1, cluster_B=4, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=1, cluster_B=5, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=1, cluster_B=6, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=1, cluster_B=7, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)
                    
plot_transition_zone(cluster_A=2, cluster_B=3, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=2, cluster_B=4, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=2, cluster_B=5, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=2, cluster_B=6, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=2, cluster_B=7, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)
                    
plot_transition_zone(cluster_A=3, cluster_B=4, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=3, cluster_B=5, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)                    

plot_transition_zone(cluster_A=3, cluster_B=6, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=3, cluster_B=7, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)                    

plot_transition_zone(cluster_A=4, cluster_B=5, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=5, cluster_B=7, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=6, cluster_B=7, node_cluster=node_cluster, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_hel=X_hel, 
                    df2=df2, variables_of_interest=sp_vars, 
                    som_shape=som_shape)            
# ~ '''
                    
# ----------------------------------------------------------------------
# Identify transition samples between all clusters
# ----------------------------------------------------------------------

# kmeans cluster color assignment:
tab20 = plt.get_cmap('tab20')
custom_indices = [0, 5, 10, 15, 19, 14, 3, 7, 9, 11, 13, 17, 2, 4]
discrete_color = [tab20(i) for i in custom_indices]
discrete_tab20 = ListedColormap(discrete_color) # usage: discrete_tab20() i.e. cluster_color(i)/(c1)/(c2)

cluster_ids = np.unique(node_cluster)
cluster_colors = {cluster: tab20(custom_indices[cluster]) for cluster in cluster_ids} # usage: cluster_colors[] i.e. cluster_color[i]/[c1]/[c2]

# create transition zone plot
plt.figure(figsize=(14, 12))
ax = plt.gca()
ax.set_aspect('equal')

# background U-Matrix
plt.imshow(umatrix, cmap='bone_r', origin='lower', alpha=0.7)

# plot cluster boundaries
for i in range(node_cluster.shape[0]):
    for j in range(node_cluster.shape[1]):
        current = node_cluster[i, j]
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + dx, j + dy
            if 0 <= ni < node_cluster.shape[0] and 0 <= nj < node_cluster.shape[1]:
                neighbor = node_cluster[ni, nj]
                if neighbor != current:
                    plt.plot([i, ni], [j, nj], 'k-', linewidth=1.5, alpha=0.5)

# transition markers
transition_pairs = find_transition_pairs(node_cluster)
edge_colors = plt.cm.tab10(np.linspace(0, 1, len(transition_pairs)))

for pair_idx, (c1, c2) in enumerate(transition_pairs):
    indices = []
    for idx, (x, y) in enumerate(bmu_locations):
        neighbors = get_neighboring_clusters((x, y), node_cluster)
        if c1 in neighbors and c2 in neighbors:
            indices.append(idx)
            
                   
    blend_color = np.mean([discrete_tab20(c1), discrete_tab20(c2)], axis=0)
    for idx in indices:
        x, y = bmu_locations[idx]
        plt.scatter(x, y, marker='o',  # markers
                   color=blend_color, 
                   s=50,               # marker size
                   alpha=0.8,
                   edgecolor=edge_colors[pair_idx],  # edge color per transition pair
                   linewidth=1.5)      # thickniess for marker border
                   

# add cluster centroids
for cluster_id, center in cluster_centers.items():
    plt.scatter(center[0], center[0], s=300, marker='',
               color=discrete_tab20(cluster_id), edgecolor='black',
               label=f'cluster {cluster_id} center')

plt.title('Transition Zones Between SOM Clusters', fontsize=14, pad=20)
plt.xlabel('SOM X Dimension', fontsize=12)
plt.ylabel('SOM Y Dimension', fontsize=12)
plt.xticks(xrange, xrange)
plt.yticks(yrange, yrange)

# legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', 
          label=f'{c1}-{c2} Transition',
          markerfacecolor='lightgray',
          markeredgecolor=edge_colors[i],
          markersize=10) for i, (c1, c2) in enumerate(transition_pairs)
] + [
    Line2D([0], [0], marker='*', color='w', 
          label=f'Cluster {i} Center',
          markerfacecolor=discrete_tab20(i),
          markersize=15) for i in cluster_centers
]

plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
           loc='upper left', framealpha=1)

plt.tight_layout()
for path in analysis_dirs:
    try:
        file_path = os.path.join(path, "transition_zones.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Simplified transition visualization saved to: {file_path}")
    except Exception as e:
        print(f"Error saving to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

# ----------------------------------------------------------------------
# SOM node weights PCA
# ----------------------------------------------------------------------
flat_weights = som.get_weights().reshape(-1, som.get_weights().shape[-1])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(flat_weights)
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['Cluster'] = node_cluster.ravel()

plt.figure(figsize=(10, 8))
for i in range(n_clusters):
    subset = pca_df[pca_df['Cluster'] == i]
    plt.scatter(subset['PC1'], subset['PC2'], 
               color=discrete_tab20(i), 
               label=f'Cluster {i}', 
               alpha=1,
               edgecolor='white',
               linewidth=0.75)

# add cluster vectors
for i in range(n_clusters):
    if i in cluster_vectors:
        vec = pca.transform([cluster_vectors[i]])[0]
        plt.arrow(0, 0, vec[0]*0.8, vec[1]*0.8, 
                 head_width=0.1, head_length=0.1, 
                 fc=discrete_tab20(i), ec=discrete_tab20(i), 
                 linestyle='-', alpha=0.5)

plt.title('PCA of SOM Node Weights', fontsize=14, pad=20)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()

for path in analysis_dirs:
    try:
        file_path = os.path.join(path, "pca_clusters.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"PCA visualization saved to: {file_path}")
    except Exception as e:
        print(f"Error saving to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

print("..... \n.... \n... \n.. \n. \n \nAnalysis complete!")

