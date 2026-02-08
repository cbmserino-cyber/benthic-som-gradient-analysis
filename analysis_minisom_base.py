## =====================================================================
## Mini Self-Organizing Map (SOM) - COVER-BASED
## objective: Analyze trained SOM using benthic cover (not per site) to uncover gradients in composition space and link them to environmental gradient
## output: U-Matrix, component planes, node clustering (KMeans)
## input features: sp covers
'''clustering cover gradients -- understanding relationship with spatial and biological patterns, interpret chemical/environmental influences on benthic structure'''
## =====================================================================

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
    "df1": ["final_dataset", "cc.csv"],
    "df2": ["final_dataset", "clustered_dataset.csv"],
    "df3": ["final_dataset", "labels.csv"]
}
    
# -- load dataset 'cover&chem.csv'--
def load_dataset(base_path, relative_path):
    full_path = os.path.join(base_path, *relative_path)
    return pd.read_csv(full_path)
    
df1 = load_dataset(base_paths["dropbox"], files["df1"])
df2 = load_dataset(base_paths["dropbox"], files["df2"])
df3 = load_dataset(base_paths["dropbox"], files["df3"])

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
print("Available base directories:", result_dirs)

# ----------------------------------------------------------------------
# Analysis path with n_clusters
# ----------------------------------------------------------------------
n_clusters=6
def make_save_dirs(save_paths, 
                   phase="1phase", 
                   iterations=500, 
                   dimension="11x11",
                   model_name="cosine_som_sp11x11",
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
    model_name="cosine_som_sp",
    n_clusters=6
)
print("Available analysis directories:", analysis_dirs)

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
    'WT (℃)', 'Zn (mg/L)', 'pH'
]

major_cats = ['Actiniaria', 'Artificial Debris', 'Ascidian', 'Black coral', 'CCA', 'Corallimorpharia', 'Fish', 'Gorgonian', 'Hard coral', 'Hydrozoa', 'Macroalgae', 'Other mobile invertebrates', 'Other sessile invertebrates', 'Seagrass', 'Soft coral', 'Sponge', 'Stable substrate', 'TWS', 'Turf', 'Unstable substrate', 'Zoanthid']

nonbio_vars = ['fistws', 'nettws', 'shatws', 'taptws', 'unktws', 'wantws']

kmeans_pca = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "cluster"]

# ~ sp_vars = [col for col in df2.columns if col not in chem_vars + env_vars + major_cats + nonbio_vars + kmeans_pca + ['site', 'region', 'SD', 'depth', 'transect', 'nlabels', 'latitude', 'longitude', 'Shannon', 'Pielou', 'BC']]

sp_vars = [col for col in df2.columns if col not in chem_vars + major_cats + kmeans_pca + ['year', 'site', 'region', 'SD', 'depth', 'transect', 'nlabels', 'latitude', 'longitude', 'Shannon', 'Pielou', 'BC']]


'''select features to train'''
# ----------------------------
variables_of_interest = sp_vars

# ----------------------------------------------------------------------
# Data preprocessing
# ----------------------------------------------------------------------
def preprocess_data(df, features):
    X = df[features].fillna(0)
    
    # log-transform skewed variables
    for var in X.columns:
        if X[var].min() >= 0:  #log-transform non-negative variables
            X[var] = np.log1p(X[var])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

# -------------------------+
# Scale feature to train   |
# -------------------------+
X_scaled, scaler = preprocess_data(df2, variables_of_interest)

# ----------------------------------------------------------------------
# Load saved SOM model
# ----------------------------------------------------------------------
output_path = os.path.join(result_dirs[0], "som_training_output.pkl")
def load_training_output(output_path):
    with open(output_path, 'rb') as f:
        data = pickle.load(f)
    
    # list required keys from trained model
    required_keys = ['som', 'X_scaled', 'df', 'variables_of_interest',
                   'bmu_sp_avg']
    
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key in training output: {key}")
    
    return data

try:
    training_data = load_training_output(output_path)
    
    # unpack training output
    som = training_data['som']
    X_scaled = training_data['X_scaled']
    # ~ df2 = training_data['df']
    variables_of_interest = training_data['variables_of_interest']
    bmu_sp_avg = training_data['bmu_sp_avg']
    # ~ bmu_env_avg = training_data['bmu_env_avg']
    # ~ node_clusters = training_data['node_clusters']
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
    bmu_locations = np.array([som.winner(x) for x in X_scaled])

try:
    X_scaled = np.load(os.path.join(result_dirs[0], "X_scaled.npy"))
except Exception as e:
    print(f"Error loading scaled data: {str(e)}")


def neighborhood_preservation_index(X, som_coords, k=5):
    """
    Compute how well SOM preserves k-nearest neighbor relationships.
    """
    nbrs_data = NearestNeighbors(n_neighbors=k).fit(X)
    _, idx_data = nbrs_data.kneighbors(X)

    nbrs_som = NearestNeighbors(n_neighbors=k).fit(som_coords)
    _, idx_som = nbrs_som.kneighbors(som_coords)

    overlaps = [len(set(idx_data[i]) & set(idx_som[i])) / k for i in range(len(X))]
    return np.mean(overlaps)

som_coords = np.array([som.winner(x) for x in X_scaled])
npi_score = neighborhood_preservation_index(X_scaled, som_coords, k=5)
print(f"Neighborhood Preservation Index: {npi_score:.3f}")
   
# ----------------------------------------------------------------------
# U-Matrix
# ----------------------------------------------------------------------
# ~ def visualize_som(som, features):
    # ~ """SOM visualizations"""
    # ~ # U-Matrix
    # ~ plt.figure(figsize=(10, 10))
    # ~ plt.pcolor(som.distance_map().T, cmap='bone_r')
    # ~ plt.colorbar()
    # ~ plt.title('SOM U-Matrix (Mahalanobis Distance)')
    # ~ for path in analysis_dirs:
        # ~ try:
            # ~ '''check if directory exists'''
            # ~ os.makedirs(path, exist_ok=True)
            # ~ file_path = os.path.join(path, "u_matrix.png")
            # ~ plt.savefig(file_path, dpi=300)
            # ~ print(f"Plot saved successfully to: {file_path}")
        # ~ except Exception as e:
            # ~ print(f"Failed to save plot to {path}: {e}")
    # ~ plt.show()
    # ~ plt.close()
    
    # ~ '''component planes'''
    # ~ num_features = len(features)
    # ~ n_cols = 4 
    # ~ n_rows = math.ceil(num_features / n_cols)

    # ~ plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    # ~ for i, var in enumerate(features):
        # ~ plt.subplot(n_rows, n_cols, i + 1)
        # ~ plt.pcolor(som.get_weights()[:, :, i].T, cmap='viridis')
        # ~ plt.colorbar()
        # ~ plt.title(var)
    # ~ plt.tight_layout()

    # ~ for path in analysis_dirs:
        # ~ try:
            # ~ os.makedirs(path, exist_ok=True)
            # ~ file_path = os.path.join(path, "component_planes.png")
            # ~ plt.savefig(file_path, dpi=300)
            # ~ print(f"Plot saved successfully to: {file_path}")
        # ~ except Exception as e:
            # ~ print(f"Failed to save plot to {path}: {e}")
    # ~ plt.show()
    # ~ plt.close()
# ~ visualize_som(som, variables_of_interest)


def visualize_som(som, features, result_dirs):
    """SOM visualizations: U-Matrix and component planes (safe memory usage)"""
    umatrix = som.distance_map()
    plt.figure(figsize=(12, 10))
    plt.imshow(umatrix, cmap='bone_r', origin='lower')
    plt.colorbar(label="Distance")
    plt.title('SOM U-Matrix (cosine Distance)')
    for path in analysis_dirs:
        try:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, "u_matrix.png")
            plt.savefig(file_path, dpi=150)
            print(f"U-Matrix saved to: {file_path}")
        except Exception as e:
            print(f"Failed to save U-Matrix to {path}: {e}")
    plt.close()
    
'''
    # --- Component Planes ---
    for i, var in enumerate(features):
        plt.figure(figsize=(5, 5))
        plt.pcolor(som.get_weights()[:, :, i].T, cmap='viridis')
        plt.colorbar()
        plt.title(f"Component Plane: {var}")
        plt.tight_layout()

        for path in result_dirs:
            try:
                os.makedirs(path, exist_ok=True)
                file_path = os.path.join(path, f"component_plane_{var}.png")
                plt.savefig(file_path, dpi=150)
                print(f"Component plane for {var} saved to: {file_path}")
            except Exception as e:
                print(f"Failed to save component plane for {var} to {path}: {e}")
        plt.close()
'''
visualize_som(som, variables_of_interest, result_dirs)

# ----------------------------------------------------------------------
# Clustering and spatial analysis
# ----------------------------------------------------------------------

def cluster_som_nodes(som, n_clusters=n_clusters):
    """Cluster SOM nodes using K-Means"""
    weights = som.get_weights().reshape(-1, X_scaled.shape[1])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(weights)
    return kmeans.labels_.reshape(som._weights.shape[0], som._weights.shape[1])

node_clusters = cluster_som_nodes(som)

# assign clusters to original data
def assign_clusters(som, data, node_clusters):
    """assign each sample to its cluster"""
    assignments = []
    for x in data:
        winner = som.winner(x)
        assignments.append(node_clusters[winner])
    return assignments

df2['SOM_Cluster'] = assign_clusters(som, X_scaled, node_clusters)

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
    
som_env = environmental_gradient_analysis(df2, 'SOM_Cluster', 'cosine', analysis_dirs)

# ----------------------------------------------------------------------
# SP vars gradient corr
# ----------------------------------------------------------------------
def sp_gradient_analysis(df, cluster_col, method_name, result_dirs=None):
    print(f"\n🎯 ASSEMBLAGE ANALYSIS: {method_name}")
    print(f"Cluster column: {cluster_col}")
    
    available_sp_vars = [var for var in sp_vars if var in df.columns]
    print(f"Analyzing {len(available_sp_vars)} assemblage variables")
    
    if not available_sp_vars:
        print("❌ No assemblage variables found!")
        return None
    
    # calculate cluster means
    cluster_means = df.groupby(cluster_col)[available_sp_vars].mean()
    
    # compute correlations
    corr_results = {}
    for var in available_sp_vars:
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
    plt.title(f'{method_name}\nAssemblage Gradient Correlations')
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
                plt.savefig(os.path.join(path, f"sp_gradients_{method_name}.png"), 
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
    
som_env = sp_gradient_analysis(df2, 'SOM_Cluster', 'cosine', analysis_dirs)

# ----------------------------------------------------------------------
# Save final clustered som node 
# ----------------------------------------------------------------------
try:
    final_path = os.path.join(analysis_dirs, "df2_with_clusters.csv")
    df2.to_csv(final_path, index=False)
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
unique_clusters = np.unique(node_clusters)
cluster_ids = np.unique(node_clusters)
n_clusters = len(cluster_ids)

# kmeans cluster color assignment:
tab20 = plt.get_cmap('tab20')
custom_indices = [0, 5, 10, 15, 19, 14]
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
# SOM node cluster clustering (KMeans)
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(
    node_clusters,
    cmap=discrete_tab20,
    annot=True,
    fmt="d"
)
plt.title('cosine-based SOM Node Clusters')
for path in analysis_dirs:
    try:
        '''checkk if directory exists'''
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "som_node_clusters.png")
        plt.savefig(file_path, dpi=300)
        print(f"Plot saved successfully to: {file_path}")
    except Exception as e:
        print(f"Failed to save plot to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

# assign clusters to original data'''
def assign_clusters(som, data, node_clusters):
    """assign each sample to its cluster"""
    assignments = []
    for x in data:
        winner = som.winner(x)
        assignments.append(node_clusters[winner])
    return assignments

df2['SOM_Cluster'] = assign_clusters(som, X_scaled, node_clusters)

ari = adjusted_rand_score(df2['cluster'], df2['SOM_Cluster'])
nmi = normalized_mutual_info_score(df2['cluster'], df2['SOM_Cluster'])
print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}")

# ---------------------------------------------------------------------
# Contingency matrix
# ---------------------------------------------------------------------
'''
shows how many samples fall into each combination of sample-level and SOM-node-level clusters

conf_mat = pd.crosstab(df2['cluster'], df2['SOM_Cluster'])
print(conf_mat)

# ----------------------------------------------------------------------
# Optimal label matching (Hungarian algorithm)
# ----------------------------------------------------------------------
from scipy.optimize import linear_sum_assignment
cm = confusion_matrix(df2['cluster'], df2['SOM_Cluster']) # confusion matrix
row_ind, col_ind = linear_sum_assignment(-cm) # convert to a cost matrix for matching
label_map = {col: row for row, col in zip(row_ind, col_ind)} # mapping


ari = adjusted_rand_score(df2['cluster'], df2['SOM_Cluster'])
nmi = normalized_mutual_info_score(df2['cluster'], df2['SOM_Cluster'])
print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}")
'''
# ----------------------------------------------------------------------
# Enhanced node cluster assignments heatmap
# ----------------------------------------------------------------------
'''visualizes node cluster assignments in 2D heatmap'''
plt.figure(figsize=(12, 10))
sns.heatmap(
    node_clusters,
    cmap=discrete_tab20,
    annot=True,
    fmt="d",
    cbar_kws={
        'label': 'Cluster ID',
        'ticks': sorted_clusters
    },
    linewidths=0.5,
    linecolor='white'
)
plt.title('SOM Node Clusters (KMeans)', fontsize=14, pad=20)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)

for path in analysis_dirs:
    try:
        file_path = os.path.join(path, "som_node_clusters1.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Cluster visualization saved to: {file_path}")
    except Exception as e:
        print(f"Error saving to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

# ----------------------------------------------------------------------
# save final clustered som node dataframe
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

    plt.title('Spatial Distribution of cosine-based SOM Clusters - Taiwan')
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

# Method 2: From Taiwan shapefile
try:
    taiwan_shapefile = "/home/bernard/Documents/thesis/taiwan_boundary"
    taiwan = gpd.read_file(taiwan_shapefile)

    # identify shapefile extent (bounding box)
    minx, miny, maxx, maxy = taiwan.total_bounds

    # manual boundary setting
    minx, maxx = 116.2, 122.3
    miny, maxy = 20.1, 26.0

    plt.figure(figsize=(12, 9))
    ax = plt.gca()

    # plot Taiwan boundary
    taiwan.plot(ax=ax, color='lightgray', edgecolor='gray', alpha=0.6)

    # overlay species richness
    sc = ax.scatter(df2['longitude'], df2['latitude'], 
                    c=df2['SOM_Cluster'], 
                    cmap=discrete_tab20, 
                    s=50, alpha=0.8, edgecolor='black', linewidth=0.5)
    # zoom extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    plt.title('Spatial Distribution of cosine-based SOM Clusters - Taiwan')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    unique_clusters = sorted(df2['SOM_Cluster'].unique())
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=cluster_colors[cluster], 
                                  markersize=8, label=f'Cluster {cluster}') 
                      for cluster in unique_clusters]
    
    # add legend
    ax.legend(handles=legend_elements, title='SOM Clusters', 
              loc='upper left', framealpha=0.9)

    plt.tight_layout()

except Exception as e:
    print(f"Shapefile not available, using simple scatter: {e}")
    plt.figure(figsize=(12, 9))
    sns.scatterplot(data=df2, x='longitude', y='latitude', hue='SOM_Cluster', 
                    palette=discrete_color, s=50, alpha=0.8)
    plt.title('Spatial Distribution of cosine-based SOM Clusters')
    
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

# ----------------------------------------------------------------------
''' biodiversity analysis '''
# ----------------------------------------------------------------------
# calculate species richness
df2['Species_Richness'] = df2[sp_vars].gt(0).sum(axis=1)

# richness vs. SOM clusters
plt.figure(figsize=(12, 6))
sns.boxplot(data=df2, x='SOM_Cluster', y='Species_Richness', palette=discrete_color)
plt.title('Species Richness Across cosine-based SOM Clusters')
plt.xlabel('SOM Cluster')
plt.ylabel('Species Richness')
for path in analysis_dirs:
    try:
        file_path = os.path.join(path, "richness_by_cluster.png")
        plt.savefig(file_path, dpi=300)
        print(f"Saved: {file_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

# Method 1: GeoPandas with natural earth data
plt.figure(figsize=(12, 9))
ax = plt.gca()
# create a GeoDataFrame from points
geometry = [Point(xy) for xy in zip(df2['longitude'], df2['latitude'])]
gdf = gpd.GeoDataFrame(df2, geometry=geometry, crs="EPSG:4326")

# plot the points
sc = ax.scatter(df2['longitude'], df2['latitude'], c=df2['Species_Richness'], 
                    cmap='turbo', s=50, alpha=0.9, linewidth=0.3)

# add basemap
ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)

# add colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Species Richness')

plt.title('Spatial Distribution of Species Richness - Taiwan')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

for path in analysis_dirs:
    try:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "species_richness_taiwan_geomap.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Map plot saved successfully to: {file_path}")
    except Exception as e:
        print(f"Failed to save plot to {path}: {e}")

plt.show(block=False)
plt.pause(1)
plt.close()

# Method 2: SHAPEFILE spatial view with richness
try:
    taiwan_shapefile = "/home/bernard/Documents/thesis/taiwan_boundary"
    taiwan = gpd.read_file(taiwan_shapefile)

    # identify shapefile extent (bounding box)
    minx, miny, maxx, maxy = taiwan.total_bounds

    # manual boundary setting
    minx, maxx = 116.2, 122.3
    miny, maxy = 20.1, 26.0

    plt.figure(figsize=(12, 9))
    ax = plt.gca()

    # plot Taiwan boundary
    taiwan.plot(ax=ax, color='lightgray', edgecolor='gray', alpha=0.6)

    #overlay species richness
    sc = ax.scatter(df2['longitude'], df2['latitude'], c=df2['Species_Richness'], 
                    cmap='turbo', s=50, alpha=0.9, linewidth=0.3)

    # Set zoom extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    plt.title('Spatial Distribution of Species Richness - Taiwan', fontsize=13)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # colorbar for Species Richness
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Species Richness')

    plt.tight_layout()

except Exception as e:
    print(f"Shapefile not available, using simple scatter: {e}")
    plt.figure(figsize=(7.5, 9))
    sc = plt.scatter(df2['longitude'], df2['latitude'], c=df2['Species_Richness'],
                    cmap='viridis', s=100, alpha=0.85)
    plt.colorbar(sc, label='Species Richness')
    plt.title('Spatial Distribution of Species Richness')

for path in analysis_dirs:
    try:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "species_richness_taiwan_shmap.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Saved species richness map to: {file_path}")
    except Exception as e:
        print(f"Failed to save plot to {path}: {e}")

plt.show(block=False)
plt.pause(1)
plt.close()   

som_x, som_y = 21, 20

'''
# (1) prepare containers for species map overlays
species_map = defaultdict(lambda: np.zeros((som_x, som_y)))
species_count = defaultdict(lambda: np.zeros((som_x, som_y)))
'''

# (2) prelocated species map
species_map = [np.zeros((som_x, som_y)) for _ in range(len(major_cats))]
neuron_counts = np.zeros((som_x, som_y)) # create single array to track sample count per neuron

'''
# (1) aggregate species data to BMU locations
for idx, (i, j) in enumerate(bmu_locations):
    for species_idx, sp_val in enumerate(df2[major_cats].iloc[idx]):
        # ~ species_map[species_idx][i, j] += sp_val
        species_map = [np.zeros((som_x, som_y)) for _ in range(len(species_idx))]
        species_count[species_idx][i, j] += 1

'''# (2) aggregate species data to BMU locations
for idx, (i, j) in enumerate(bmu_locations):
    neuron_counts[i, j] += 1
    row = df2[major_cats].iloc[idx].values
    for k in range(len(major_cats)):
        species_map[k][i, j] += row[k]
'''

# (1) compute average cover per SOM unit
for species_idx in species_map:
    with np.errstate(divide='ignore', invalid='ignore'):
        species_map[species_idx] = np.divide(
            species_map[species_idx], 
            species_count[species_idx],
            out=np.zeros_like(species_map[species_idx]),
            where=species_count[species_idx] != 0
        )
'''
# (2) compute average cover per SOM unit per species
for k in range(len(major_cats)):
    species_map[k] = np.divide(
        species_map[k], 
        neuron_counts, 
        out=np.zeros_like(species_map[k]),
        where=neuron_counts != 0
    )
'''

# (1) plot overlay for first nth species
fig, axes = plt.subplots(4, 5, figsize=(18, 10))
for i, ax in enumerate(axes.flat):
    im = ax.imshow(species_map[i].T, origin='lower', cmap='viridis')
    ax.set_title(major_cats[i])
    fig.colorbar(im, ax=ax)
plt.tight_layout()
for path in analysis_dirs:
    try:    
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "Species_overlay.png")
        plt.savefig(file_path, dpi=300)
        print(f"Plot saved successfully to: {file_path}")
    except Exception as e:
            print(f"Failed to save plot to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

'''
# (2) plot overlay for species
fig, axes = plt.subplots(4, 5, figsize=(18, 10))
for i, ax in enumerate(axes.flat):
    if i < len(major_cats):
        im = ax.imshow(species_map[i], origin='lower', cmap='viridis')
        ax.set_title(major_cats[i])
        fig.colorbar(im, ax=ax)
    else:
        ax.axis('off')  # Turn off extra axes
plt.tight_layout()
for path in analysis_dirs:
    try:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "Species_overlay.png")
        plt.savefig(file_path, dpi=300)
        print(f"Plot saved successfully to: {file_path}")
    except Exception as e:
            print(f"Failed to save plot to {path}: {e}")
# ~ plt.show()
    plt.close()

# ----------------------------------------------------------------------
# Species composition heatmap by SOM cluster
# ----------------------------------------------------------------------
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
    df=df2_cluster, 
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
    for cluster_idx in range(n_clusters):
        contrib = {}
        for feature in feature_cols:
            category = labels_dict[feature]
            contrib[category] = contrib.get(category, 0) + cluster_means.loc[cluster_idx, feature]
            
        contrib = {k: v * 100 for k, v in contrib.items() if v > 0} # convert to percentages
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
        axs[i].set_title(f'Cluster {i} Composition')
        
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
    for cluster_idx in range(n_clusters):
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
            axs[i].set_title(f'Cluster {i}\nTop 5 Categories', fontweight='bold', fontsize=12)
            
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
            axs[i].set_title(f'Cluster {i}\n(No Data)')
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
    for cluster_idx in range(n_clusters):
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
            wedges, texts = axs[i].pie(vals, colors=colors, startangle=140)
            axs[i].set_title(f'Cluster {i}\nTop 5 Categories', fontweight='bold', fontsize=12)
            
            # legend
            legend_labels = []
            for j, (category, value) in enumerate(zip(cats, vals)):
                if category == 'Others' and others_percentage > 0:
                    legend_labels.append(f'Others: {value:.1f}%')
                else:
                    legend_labels.append(f'{category}: {value:.1f}%')
            
            axs[i].legend(wedges, legend_labels, loc='upper left', bbox_to_anchor=(1, 0, 0.5, 1),
                         title="Categories", title_fontsize=10)
        else:
            axs[i].set_title(f'Cluster {i}\n(No Data)')
            axs[i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axs[i].transAxes)
    
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
    df2=df2_cluster,
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

# ~ plot_cluster_major_category_bars(
    # ~ cluster_contributions,
    # ~ label_map=df3,
    # ~ analysis_dirs=analysis_dirs
# ~ )

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

# ~ plot_cluster_major_category_bars(
    # ~ cluster_contributions,
    # ~ label_map=df3,
    # ~ analysis_dirs=analysis_dirs
# ~ )

# ----------------------------------------------------------------------
# Overlay sites on U-Matrix
# ----------------------------------------------------------------------
def overlay_site_names_on_som(som, df, label_col='site'):
    umatrix = som.distance_map()
    plt.figure(figsize=(12, 10))
    plt.imshow(umatrix, cmap='bone_r', origin='lower')  # SAME orientation
    plt.colorbar(label="Distance")
    plt.title('cosine-based U-Matrix with Site Labels')

    # BMU assignment
    bmu_labels = defaultdict(list)
    for idx, row in df.iterrows():
        x, y = som.winner(X_scaled[idx])
        bmu_labels[(x, y)].append(str(row[label_col]))

    # p]lace labels at the same grid coordinates
    for (x, y), labels in bmu_labels.items():
        label_text = ", ".join(sorted(set(labels)))
        if len(label_text) > 15:
            label_text = label_text[:12] + "..."
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

overlay_site_names_on_som(som, df2, label_col='site')

def overlay_site_sample_on_som(som, df, X_scaled, label_col='site'):
    """Overlay site labels and sample counts (n=) per SOM node."""
    
    # Compute U-Matrix for background
    umatrix = som.distance_map()
    plt.figure(figsize=(12, 10))
    plt.imshow(umatrix, cmap='bone_r', origin='lower')
    plt.colorbar(label="Distance")
    plt.title("U-Matrix with Site Labels and Sample Counts")

    # Group samples by BMU coordinates
    bmu_labels = defaultdict(list)
    for idx, row in df.iterrows():
        x, y = som.winner(X_scaled[idx])
        bmu_labels[(x, y)].append(str(row[label_col]))

    # Overlay text annotations
    for (x, y), labels in bmu_labels.items():
        # unique site labels
        unique_labels = sorted(set(labels))
        label_text = ", ".join(unique_labels)
        if len(label_text) > 15:  # truncate long strings
            label_text = label_text[:12] + "..."
        
        # sample count
        count = len(labels)
        
        # combined text: site + n
        display_text = f"{label_text}\n(n={count})"
        
        plt.text(
            y, x, display_text, 
            color='red', fontsize=8, fontweight='bold',
            ha='center', va='center', 
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
        )

    plt.tight_layout()
    for path in analysis_dirs:
        file_path = os.path.join(path, f"u_matrix_overlay_{label_col}_with_counts.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Overlay plot saved: {file_path}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

overlay_site_sample_on_som(som, df2, X_scaled, label_col='site')

# ----------------------------------------------------------------------
# Overlay KMeans clusters on U-Matrix
# ----------------------------------------------------------------------
def overlay_kmeans_names_on_som(som, df, label_col='cluster'):
    umatrix = som.distance_map()
    plt.figure(figsize=(12, 10))
    plt.imshow(umatrix, cmap='bone_r', origin='lower')
    plt.colorbar(label="Distance")
    plt.title('cosine-based U-Matrix with kmeans labels')

    # BMU assignment
    bmu_labels = defaultdict(list)
    for idx, row in df.iterrows():
        x, y = som.winner(X_scaled[idx])
        bmu_labels[(x, y)].append(str(row[label_col]))

    # Place labels at the same grid coordinates
    for (x, y), labels in bmu_labels.items():
        label_text = ", ".join(sorted(set(labels)))
        if len(label_text) > 15:
            label_text = label_text[:12] + "..."
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
def overlay_kmeans_clusters_colored(som, df, X_scaled, analysis_dirs,
                                    label_col='cluster', n_clusters=5):
    """
    overlay SOM nodes with dominant KMeans cluster color
    """
    umatrix = som.distance_map()
    plt.figure(figsize=(12, 10))
    plt.imshow(umatrix, cmap='bone_r', origin='lower')
    plt.colorbar(label="Distance")
    plt.title('cosine-based U-Matrix by Dominant BMU-level KMeans Cluster')

    # fixed color pallete
    tab20 = plt.get_cmap('tab20')
    custom_indices = [0, 5, 10, 15, 19, 14]
    cluster_colors = [tab20(i) for i in custom_indices[:n_clusters]]
    cmap = ListedColormap(cluster_colors)

    # BMU assignment
    bmu_clusters = defaultdict(list)
    for idx, row in df.iterrows():
        x, y = som.winner(X_scaled[idx])
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
    X_scaled=X_scaled,
    analysis_dirs=analysis_dirs,
    label_col='cluster',
    n_clusters=n_clusters
)

def overlay_kmeans_clusters_hexagonal(som, df, X_scaled, analysis_dirs, label_col='cluster', n_clusters=5):
    """Overlay colored KMeans cluster markers on hexagonal SOM nodes."""
    umatrix = som.distance_map()
    weights = som.get_weights()
    
    plt.figure(figsize=(12, 10))
    plt.imshow(umatrix, cmap='bone_r', origin='lower', alpha=0.5)
    plt.colorbar(label="Distance")
    plt.title('cosine-based U-Matrix with KMeans Cluster Colors (Hexagonal Nodes)')

    # fixed color palette
    tab20 = plt.get_cmap('tab20')
    custom_indices = [0, 5, 10, 15, 19, 14]
    cluster_colors = [tab20(i) for i in custom_indices]

    # sssign BMUs and find dominant cluster per node
    bmu_clusters = defaultdict(list)
    for idx, row in df.iterrows():
        x, y = som.winner(X_scaled[idx])
        bmu_clusters[(x, y)].append(int(row[label_col]))
    
    # hexagonal patches for each node
    for (x, y), clusters in bmu_clusters.items():
        if clusters:
            # find dominant cluster
            cluster_counts = pd.Series(clusters).value_counts() # drops cluster labels that don’t appear in that node
            dominant_cluster = cluster_counts.index[0]
            dominant_pct = (cluster_counts.iloc[0] / len(clusters)) * 100
            
            color = cluster_colors[dominant_cluster % len(cluster_colors)]
            
            # Create hexagonal patch (adjust radius as needed)
            hexagon = RegularPolygon((y, x), numVertices=6, radius=0.5,
                                   orientation=np.pi/6,
                                   facecolor=color, alpha=0.8,
                                   edgecolor='black', linewidth=1)
            plt.gca().add_patch(hexagon)
            
            # cluster label
            plt.text(y, x, f'{dominant_cluster}', 
                    ha='center', va='center', 
                    fontsize=9, fontweight='bold',
                    color='white' if dominant_pct > 70 else 'black')

    # cluster legend
    handles = [mpatches.Patch(color=cluster_colors[i], label=f'Cluster {i}') for i in range(n_clusters)]
    plt.legend(handles=handles, title="KMeans Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    for path in analysis_dirs:
        file_path = os.path.join(path, f"u_matrix_kmeans_hexagonal_nodes.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Hexagonal nodes cluster overlay saved: {file_path}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    return bmu_clusters

overlay_kmeans_clusters_hexagonal(
    som=som,
    df= df2,
    X_scaled=X_scaled,
    analysis_dirs=analysis_dirs,
    label_col='cluster',
    n_clusters=n_clusters
)

# ----------------------------------------------------------------------
# MiniSOM node class assignment
# ----------------------------------------------------------------------

# create grid
som_shape = som.get_weights().shape[:2]
xrange = np.arange(som_shape[0])
yrange = np.arange(som_shape[1])
xx, yy = np.meshgrid(xrange, yrange)
# ~ xx, yy = np.meshgrid(xrange, yrange, indexing='ij')

f = plt.figure(figsize=(12, 10))
ax = f.add_subplot(111)
ax.set_aspect('equal')
plt.title('SOM Node Class Composition', pad=20, fontsize=14)

# kmeans cluster color assignment:
tab20 = plt.get_cmap('tab20')
custom_indices = [0, 5, 10, 15, 19, 14]
discrete_colors = [tab20(i) for i in custom_indices]
discrete_tab20 = ListedColormap(discrete_colors)

cluster_ids = np.unique(node_clusters)
labels = df2['cluster'].values
cluster_colors = {cluster: discrete_colors[cluster] for cluster in cluster_ids}

# create winner mapping with cluster composition
winmap = defaultdict(list)
for i, x in enumerate(X_scaled):
    winner = som.winner(x)
    cluster_label = labels[i]  # cluster assignments
    winmap[winner].append(cluster_label)

# calculate class proportions for each node
class_counts = {pos: Counter(lbls) for pos, lbls in winmap.items()}

# unique labels for color mapping
unique_labels = sorted(np.unique(labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

# pie charts for each node
for (i, j), counter in class_counts.items():
    total = sum(counter.values())
    if total == 0:
        continue
    
    # center position and hexagonal spacing
    center_x = i
    center_y = j * np.sqrt(3) / 2
    
    # pie chart radius
    radius = 0.4 * min(1, np.log(total + 1))
    
    # pie wedges
    start_angle = 90  # start from top
    for label, count in counter.items():
        angle = 360 * count / total
        
        # Use your cluster color assignment
        if label in cluster_colors:
            color = cluster_colors[label]
        else:
            # Fallback for unexpected labels
            color = 'lightgray'
        
        # wedge with clean edges
        wedge = Wedge(
            (center_x, center_y), 
            radius, 
            start_angle, 
            start_angle + angle,
            facecolor=color, 
            edgecolor='white', 
            linewidth=0.8,
            alpha=0.9
        )
        ax.add_patch(wedge)
        start_angle += angle
    
    # node coordinates text
    ax.text(
        center_x, center_y, 
        f'({i},{j})', 
        ha='center', 
        va='center', 
        fontsize=6, 
        color='black',
        alpha=0.5
    )

# legend using your cluster colors
legend_elements = [
    Patch(
        facecolor=cluster_colors[label],
        edgecolor='white',
        label=f'Cluster {label} (n={(np.array(labels) == label).sum()})'
    ) for label in sorted(cluster_colors.keys())
]

ax.legend(
    handles=legend_elements,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    title="Cluster Composition",
    framealpha=1
)

# adjust grid labels
plt.xticks(xrange, [str(i+1) for i in xrange])
plt.yticks(yrange * np.sqrt(3) / 2, [str(i+1) for i in yrange])
plt.xlabel('SOM X Dimension', labelpad=10)
plt.ylabel('SOM Y Dimension', labelpad=10)

# grid lines
for i in xrange:
    for j in yrange:
        ax.plot(i, j * np.sqrt(3)/2, 'o', markersize=2, color='gray', alpha=0.2)

# save
for path in analysis_dirs:
    try:
        file_path = os.path.join(path, "som_class_assignment_pie.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Class assignment visualization saved to: {file_path}")
    except Exception as e:
        print(f"Error saving to {path}: {e}")

plt.tight_layout()
plt.show(block=False)
plt.pause(1)
plt.close()

# ----------------------------------------------------------------------
# Classification fxn
# ----------------------------------------------------------------------
"""classify samples using the label assigned to the associated winning neuron"""
def classify(som, data, winmap, default_class=None):
    # classify samples using pre-computed winmap with safety checks
    """a label 'A' is associated to neuron if the majority of samples mapped in that neuron have label 'A' """
    """assign the most common label in the dataset in case that a sample is mapped to a neuron for which no class is assigned"""
    if not winmap:
        raise ValueError("winmap cannot be empty")
    
    if default_class is None:
        default_class = max(sum(winmap.values(), Counter()).items(), 
                          key=lambda x: x[1])[0]
    
    return [winmap.get(som.winner(d), Counter({default_class: 1})).most_common(1)[0][0] 
            for d in data]
'''
def visualize_classification(som, X, y_true, winmap, analysis_dirs):
    """creates classification visualizations"""
    # 1. confusion matrix
    y_pred = classify(som, X, winmap)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                 display_labels=np.unique(y_true))
    disp.plot(cmap='Blues', ax=plt.gca(), values_format='d')
    plt.title('SOM Classification Performance', pad=20)
    for path in analysis_dirs:
        try:
            file_path = os.path.join(path, "confusion_matrix.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"confusion matrix plot saved: {file_path}")
        except Exception as e:
            print(f"Error saving error confusion matrix plot: {e}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    # 2. SOM node purity plot
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    ax.set_aspect('equal')
    
    # calculate node purity
    node_purities = {}
    for pos, counts in winmap.items():
        total = sum(counts.values())
        node_purities[pos] = counts.most_common(1)[0][1] / total
    
    # create hexagonal grid
    som_shape = (som._weights.shape[0], som._weights.shape[1])
    hex_radius = 0.5 / np.sin(np.pi/3)
    
    for i in range(som_shape[0]):
        for j in range(som_shape[1]):
            if (i,j) not in winmap:
                continue
                
            x = i + (0.5 if j % 2 == 1 else 0)
            y = j * np.sqrt(3)/2
            dominant_class = winmap[(i,j)].most_common(1)[0][0]
            purity = node_purities[(i,j)]
            
            hexagon = RegularPolygon(
                (x, y), 
                numVertices=6, 
                radius=hex_radius,
                facecolor=plt.cm.tab20(dominant_class % 20),
                alpha=purity**0.5,
                edgecolor='white'
            )
            ax.add_patch(hexagon)
            
            plt.text(x, y, f"{dominant_class}\n({purity:.0%})", 
                    ha='center', va='center', fontsize=8)

    plt.title('SOM Node Purity by Dominant Class', pad=20)
    
    # colorbar implementation
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap='viridis_r')
    sm.set_array([])
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(sm, cax=cax, label='Node Purity (Darker = More Pure)')
    
    for path in analysis_dirs:
        try:
            file_path = os.path.join(path, "node_purity.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"node purity plot saved: {file_path}")
        except Exception as e:
            print(f"Error saving node purity plot: {e}")
    plt.close()
'''

def visualize_classification(som, X, y_true, winmap, analysis_dirs):
    """Simplified version to reduce memory usage"""
    
    # 1. confusion Matrix
    y_pred = classify(som, X, winmap)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                 display_labels=np.unique(y_true))
    disp.plot(cmap='Blues', ax=plt.gca(), values_format='d')
    plt.title('SOM Classification Performance', pad=20)
    
    for path in analysis_dirs:
        try:
            file_path = os.path.join(path, "confusion_matrix.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"confusion matrix plot saved: {file_path}")
        except Exception as e:
            print(f"Error saving confusion matrix: {e}")
    plt.close()
    
    # 2. node Purity Plot
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    ax.set_aspect('equal')
    
    som_shape = (som._weights.shape[0], som._weights.shape[1])
    
    # scatter plot instead of polygons
    x_coords, y_coords, colors, sizes = [], [], [], []
    
    for i in range(som_shape[0]):
        for j in range(som_shape[1]):
            if (i,j) not in winmap:
                continue
                
            counts = winmap[(i,j)]
            total = sum(counts.values())
            purity = counts.most_common(1)[0][1] / total
            dominant_class = counts.most_common(1)[0][0]
            
            x_coords.append(i)
            y_coords.append(j)
            colors.append(dominant_class)
            sizes.append(100 + 900 * purity)  # Scale size by purity
    
    scatter = ax.scatter(x_coords, y_coords, c=colors, s=sizes, 
                        cmap='tab20', alpha=0.7, edgecolors='white')
    
    # labels for high-purity nodes only
    for i, j in winmap.keys():
        counts = winmap[(i,j)]
        purity = counts.most_common(1)[0][1] / sum(counts.values())
        if purity > 0.7:  # Only label high-purity nodes
            ax.text(i, j, f"{counts.most_common(1)[0][0]}", 
                   ha='center', va='center', fontsize=8)
    
    plt.title('SOM Node Purity (Size = Purity, Color = Dominant Class)', pad=20)
    plt.colorbar(scatter, label='Dominant Class')
    
    for path in analysis_dirs:
        try:
            file_path = os.path.join(path, "node_purity_simplified.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Simplified node purity plot saved: {file_path}")
        except Exception as e:
            print(f"Error saving simplified plot: {e}")
    plt.close()      

# main execution:
if __name__ == "__main__":
    # load pre-trained data
    try:
        output_path = os.path.join(result_dirs[0], "som_training_output.pkl")
        training_data = load_training_output(output_path)
        
        som = training_data['som']
        X_scaled = training_data['X_scaled']
        variables_of_interest = training_data['variables_of_interest']
        bmu_locations = training_data['bmu_locations']
        
    except Exception as e:
        print(f"Error loading training output: {e}")
        sys.exit(1)

    # prepare data
    data = X_scaled
    labels = df2['SOM_Cluster'].values
    
    # train-test split - using same random_state as during training
    X_train, X_test, y_train, y_test = train_test_split(
        data, 
        labels, 
        test_size=0.3, 
        stratify=labels, 
        random_state=42  # matched with training phase!
    )
    
    # create winmap using pre-trained SOM
    winmap = defaultdict(Counter)
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        winmap[som.winner(x)][y] += 1
    
    # winmap using MiniSom's built-in:
    # ~ winmap = som.labels_map(X_train, y_train)
    
    # evaluate and visualize
    print("Classification Report:")
    print(classification_report(y_test, classify(som, X_test, winmap)))
    
    print("\nVisualizing results...")
    visualize_classification(som, X_train, y_train, winmap, analysis_dirs)
    
# ----------------------------------------------------------------------
# U-Matrix + selected species overlays
# ----------------------------------------------------------------------
def plot_species_overlays(som, df, species_list, X_scaled, n_cols=4):
    n_categories = len(species_list)
    n_panels = n_categories + 1  # +1 for U-Matrix
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Panel A: U-Matrix
    axes[0].imshow(som.distance_map(), cmap='bone_r', origin='lower')
    axes[0].set_title('cosine-based U-Matrix')

    for i, species in enumerate(species_list):
        species_density = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1]))

        for idx, row in df.iterrows():
            x, y = som.winner(X_scaled[idx])
            species_density[x, y] += row.get(species, 0)

        axes[i + 1].imshow(species_density, origin='lower', cmap='viridis')
        axes[i + 1].set_title(species)

    plt.tight_layout()
    for path in analysis_dirs:
        try:
            file_path = os.path.join(path, "species_overlay_panels.png")
            plt.savefig(file_path, dpi=300)
            print(f"Species overlay plot saved: {file_path}")
        except Exception as e:
            print(f"Error saving species overlay plot: {e}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

plot_species_overlays(som, df2, ['pescca', 'othcca', 'monfol', 'monenc', 'tursed', 'turfil', 'pocbus'], X_scaled) # check kmeans+pca for ost pronounced features
# ~ plot_species_overlays(som, df2, sp_vars, X_scaled)

# ======================================================================
# Distance analysis
# ======================================================================

# ----------------------------------------------------------------------
# Analyze site label distribution across SOM nodes
# ----------------------------------------------------------------------
def analyze_site_homogeneity(som, df, X_scaled, site_col='site'):
    """
    Analyze how site labels are distributed across SOM nodes
    Returns measures of homogeneity/heterogeneity per node
    """
    # get BMU for each sample
    bmu_sites = defaultdict(list)
    for idx, row in df.iterrows():
        x, y = som.winner(X_scaled[idx])
        bmu_sites[(x, y)].append(row[site_col])
    
    # homogeneity metrics per node
    node_stats = {}
    for (x, y), sites in bmu_sites.items():
        site_counts = pd.Series(sites).value_counts()
        total_samples = len(sites)
        unique_sites = len(site_counts)
        
        # homogeneity metrics
        if total_samples > 0:
            # most common site percentage (homogeneity measure)
            dominant_site_pct = (site_counts.iloc[0] / total_samples) * 100
            
            # Shannon diversity (variety) of sites (heterogeneity measure)
            proportions = site_counts / total_samples
            shannon_diversity = -np.sum(proportions * np.log(proportions + 1e-10))
            
            # Simpson dominance index
            simpson_dominance = np.sum(proportions ** 2)
        else:
            dominant_site_pct = 0
            shannon_diversity = 0
            simpson_dominance = 0
        
        node_stats[(x, y)] = {
            'total_samples': total_samples,
            'unique_sites': unique_sites,
            'dominant_site': site_counts.index[0] if len(site_counts) > 0 else None,
            'dominant_site_pct': dominant_site_pct,
            'shannon_diversity': shannon_diversity,
            'simpson_dominance': simpson_dominance,
            'site_counts': site_counts.to_dict()
        }
    
    return node_stats, bmu_sites

node_stats, bmu_sites = analyze_site_homogeneity(som, df2, X_scaled, site_col='site')

# plot site 
def plot_site_homogeneity_heatmap(som, node_stats):
    """
    Create heatmap showing site homogeneity across SOM nodes
    """
    umatrix = som.distance_map()
    grid_shape = umatrix.shape
    
    # create matrices for different metrics
    dominant_pct_matrix = np.zeros(grid_shape)
    diversity_matrix = np.zeros(grid_shape)
    sample_count_matrix = np.zeros(grid_shape)
    
    for (x, y), stats in node_stats.items():
        dominant_pct_matrix[x, y] = stats['dominant_site_pct']
        diversity_matrix[x, y] = stats['shannon_diversity']
        sample_count_matrix[x, y] = stats['total_samples']
    
    # subplots of heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. dominant site percentage (homogeneity)
    im1 = axes[0, 0].imshow(dominant_pct_matrix, cmap='RdYlGn', origin='lower')
    axes[0, 0].set_title('% of Dominant Site per Node')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. variety of site per node (heterogeneity)
    im2 = axes[0, 1].imshow(diversity_matrix, cmap='RdYlBu_r', origin='lower')
    axes[0, 1].set_title('Site Variety per Node \n(Shannon)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. Sample count per node
    im3 = axes[1, 0].imshow(sample_count_matrix, cmap='Purples', origin='lower')
    axes[1, 0].set_title('Sample Count per Node')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 4. Combined visualization
    combined = dominant_pct_matrix / 100  # Normalize
    im4 = axes[1, 1].imshow(combined, cmap='RdYlGn', origin='lower')
    axes[1, 1].set_title('Homogeneity (Red=Mixed, Green=Uniform)')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Add node labels with site information
    for (x, y), stats in node_stats.items():
        if stats['total_samples'] > 0:
            label = f"{stats['dominant_site_pct']:.0f}%"
            axes[0, 0].text(y, x, label, ha='center', va='center', fontsize=8, 
                           color='black' if stats['dominant_site_pct'] < 70 else 'white')
            axes[1, 1].text(y, x, stats['unique_sites'], ha='center', va='center', 
                           fontsize=8, color='black')
    
    plt.tight_layout()
    for path in analysis_dirs:
        file_path = os.path.join(path, "site_homogeneity_analysis.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Site homogeneity analysis saved: {file_path}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

plot_site_homogeneity_heatmap(som, node_stats)

# print
def print_site_homogeneity_stats(node_stats):
    """Print detailed statistics about site distribution"""
    print("=" * 60)
    print("SITE UNIFORMITY ANALYSIS ACROSS SOM NODES")
    print("=" * 60)
    
    # Overall statistics
    all_dominant_pcts = [stats['dominant_site_pct'] for stats in node_stats.values()]
    all_diversities = [stats['shannon_diversity'] for stats in node_stats.values()]
    all_samples = [stats['total_samples'] for stats in node_stats.values()]
    
    print(f"Total nodes analyzed: {len(node_stats)}")
    print(f"Total samples mapped: {sum(all_samples)}")
    print(f"Average samples per node: {np.mean(all_samples):.1f}")
    print(f"Average dominant site percentage: {np.mean(all_dominant_pcts):.1f}%")
    print(f"Average site diversity: {np.mean(all_diversities):.2f}")
    
    # Categorize nodes by homogeneity (SITE UNIFORMITY)
    homogeneous_nodes = [s for s in node_stats.values() if s['dominant_site_pct'] >= 80]
    mixed_nodes = [s for s in node_stats.values() if 50 <= s['dominant_site_pct'] < 80]
    heterogeneous_nodes = [s for s in node_stats.values() if s['dominant_site_pct'] < 50]
    
    print(f"\nHomogeneity Categories:")
    print(f"  Homogeneous nodes (≥80% one site): {len(homogeneous_nodes)}")
    print(f"  Mixed nodes (50-80% one site): {len(mixed_nodes)}")
    print(f"  Heterogeneous nodes (<50% one site): {len(heterogeneous_nodes)}")
    
    # most homogeneous nodes (nodes with uniform sites)
    print(f"\nMost Homogeneous Nodes (uniformity of sites):")
    sorted_homogeneous = sorted(node_stats.items(), 
                              key=lambda x: x[1]['dominant_site_pct'], reverse=True)[:5]
    for (x, y), stats in sorted_homogeneous:
        print(f"  Node ({x},{y}): {stats['dominant_site_pct']:.1f}% {stats['dominant_site']} "
              f"({stats['total_samples']} samples, {stats['unique_sites']} unique sites)")
    
    # most heterogeneous nodes (nodes with different sites)
    print(f"\nMost Heterogeneous Nodes (variant site per node:")
    sorted_heterogeneous = sorted(node_stats.items(), 
                                key=lambda x: x[1]['shannon_diversity'], reverse=True)[:5]
    for (x, y), stats in sorted_heterogeneous:
        print(f"  Node ({x},{y}): Diversity={stats['shannon_diversity']:.2f}, "
              f"{stats['unique_sites']} unique sites, {stats['total_samples']} samples")

print_site_homogeneity_stats(node_stats)


# ----------------------------------------------------------------------
# Analyze node diversity per site
# ----------------------------------------------------------------------
def analyze_node_diversity_per_site(som, df, site_col='site'):
    """
    analyze how each site's samples are distributed across SOM nodes
    ** measures of node diversity for each site
    *** how spread sites are distributed acroess SOM grid/topology
    """
    # BMU for each sample and group by site
    site_node_mapping = defaultdict(list)
    for idx, row in df.iterrows():
        x, y = som.winner(X_scaled[idx])
        site_node_mapping[row[site_col]].append((x, y))
    
    # calculate node diversity metrics per site
    site_stats = {}
    for site, nodes in site_node_mapping.items():
        node_counts = pd.Series(nodes).value_counts()
        total_samples = len(nodes)
        unique_nodes = len(node_counts)
        
        if total_samples > 0:
            # node distribution metrics
            proportions = node_counts / total_samples
            
            # most common node percentage (concentration measure)
            dominant_node_pct = (node_counts.iloc[0] / total_samples) * 100
            
            # Shannon diversity of nodes per site
            shannon_diversity = -np.sum(proportions * np.log(proportions + 1e-10))
            
            # Simpson dominance index
            simpson_dominance = np.sum(proportions ** 2)
            
            # node dispersion (spread across SOM)
            node_coords = list(set(nodes))  # Unique nodes
            if len(node_coords) > 1:
                # Calculate spatial spread of nodes
                x_coords = [coord[0] for coord in node_coords]
                y_coords = [coord[1] for coord in node_coords]
                spatial_spread = np.std(x_coords) + np.std(y_coords)
            else:
                spatial_spread = 0
        else:
            dominant_node_pct = 0
            shannon_diversity = 0
            simpson_dominance = 0
            spatial_spread = 0
        
        site_stats[site] = {
            'total_samples': total_samples,
            'unique_nodes': unique_nodes,
            'dominant_node': node_counts.index[0] if len(node_counts) > 0 else None,
            'dominant_node_pct': dominant_node_pct,
            'shannon_diversity': shannon_diversity,
            'simpson_dominance': simpson_dominance,
            'spatial_spread': spatial_spread,
            'node_distribution': node_counts.to_dict(),
            'node_coverage': (unique_nodes / total_samples) * 100 if total_samples > 0 else 0
        }
    
    return site_stats, site_node_mapping

site_stats, site_node_mapping = analyze_node_diversity_per_site(som, df2, site_col='site')

# plot
def plot_node_diversity_per_site(site_stats):
    """visualizations for node diversity across sites"""
    sites = list(site_stats.keys())
    
    # create summary dataframe to plot
    plot_data = []
    for site, stats in site_stats.items():
        plot_data.append({
            'site': site,
            'total_samples': stats['total_samples'],
            'unique_nodes': stats['unique_nodes'],
            'dominant_node_pct': stats['dominant_node_pct'],
            'shannon_diversity': stats['shannon_diversity'],
            'node_coverage': stats['node_coverage'],
            'spatial_spread': stats['spatial_spread']
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # (1) node diversity vs sample size
    scatter = axes[0, 0].scatter(plot_df['total_samples'], plot_df['shannon_diversity'],
                                c=plot_df['unique_nodes'], cmap='viridis', s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Total Samples')
    axes[0, 0].set_ylabel('Node Diversity (Shannon)')
    axes[0, 0].set_title('Node Diversity vs Sample Size')
    plt.colorbar(scatter, ax=axes[0, 0], label='Unique Nodes')
    
    # add site labels
    for idx, row in plot_df.iterrows():
        axes[0, 0].annotate(row['site'], (row['total_samples'], row['shannon_diversity']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # (2) identify dominant node
    sorted_sites = plot_df.sort_values('dominant_node_pct')
    axes[0, 1].barh(range(len(sorted_sites)), sorted_sites['dominant_node_pct'])
    axes[0, 1].set_yticks(range(len(sorted_sites)))
    axes[0, 1].set_yticklabels(sorted_sites['site'])
    axes[0, 1].set_xlabel('Dominant Node Percentage (%)')
    axes[0, 1].set_title('Site Concentration in Single Node\n(Lower = More Diverse)')
    
    # (3) unique nodes per site
    sorted_sites_nodes = plot_df.sort_values('unique_nodes')
    axes[0, 2].barh(range(len(sorted_sites_nodes)), sorted_sites_nodes['unique_nodes'])
    axes[0, 2].set_yticks(range(len(sorted_sites_nodes)))
    axes[0, 2].set_yticklabels(sorted_sites_nodes['site'])
    axes[0, 2].set_xlabel('Number of Unique Nodes')
    axes[0, 2].set_title('Node Diversity per Site')
    
    # (4) node coverage (unique nodes / total samples)
    sorted_coverage = plot_df.sort_values('node_coverage')
    axes[1, 0].barh(range(len(sorted_coverage)), sorted_coverage['node_coverage'])
    axes[1, 0].set_yticks(range(len(sorted_coverage)))
    axes[1, 0].set_yticklabels(sorted_coverage['site'])
    axes[1, 0].set_xlabel('Node Coverage (%)')
    axes[1, 0].set_title('Node Coverage\n(Unique Nodes / Total Samples)')
    
    # (5) spread of nodes
    sorted_spread = plot_df.sort_values('spatial_spread')
    axes[1, 1].barh(range(len(sorted_spread)), sorted_spread['spatial_spread'])
    axes[1, 1].set_yticks(range(len(sorted_spread)))
    axes[1, 1].set_yticklabels(sorted_spread['site'])
    axes[1, 1].set_xlabel('Spatial Spread Index')
    axes[1, 1].set_title('Geographic Spread of Nodes in SOM')
    
    # (6) scatter category for diversity
    colors = ['red' if x > 70 else 'orange' if x > 50 else 'green' 
             for x in plot_df['dominant_node_pct']]
    axes[1, 2].scatter(plot_df['shannon_diversity'], plot_df['unique_nodes'],
                      c=colors, s=100, alpha=0.7)
    axes[1, 2].set_xlabel('Node Diversity (Shannon)')
    axes[1, 2].set_ylabel('Unique Nodes')
    axes[1, 2].set_title('Node Diversity vs Unique Nodes\n(Colors by Concentration)')
    
    # add site labels
    for idx, row in plot_df.iterrows():
        axes[1, 2].annotate(row['site'], (row['shannon_diversity'], row['unique_nodes']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    for path in analysis_dirs:
        file_path = os.path.join(path, "node_diversity_per_site.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Node diversity per site saved: {file_path}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

plot_node_diversity_per_site(site_stats)

# node diversity statistics
def print_node_diversity_stats(site_stats):
    """Print detailed statistics about node diversity per site"""
    print("=" * 60)
    print("NODE DIVERSITY ANALYSIS PER SITE")
    print("=" * 60)
    
    # categorize sites by node diversity
    concentrated_sites = [s for s, stats in site_stats.items() if stats['dominant_node_pct'] >= 70]
    moderate_sites = [s for s, stats in site_stats.items() if 50 <= stats['dominant_node_pct'] < 70]
    diverse_sites = [s for s, stats in site_stats.items() if stats['dominant_node_pct'] < 50]
    
    print(f"Total sites analyzed: {len(site_stats)}")
    print(f"\nDiversity Categories:")
    print(f"  Concentrated sites (≥70% in one node): {len(concentrated_sites)}")
    print(f"  Moderate sites (50-70% in one node): {len(moderate_sites)}")
    print(f"  Diverse sites (<50% in one node): {len(diverse_sites)}")
    
    # most concentrated sites (low diversity)
    print(f"\nMost Concentrated Sites (Low Node Diversity):")
    sorted_concentrated = sorted(site_stats.items(), 
                               key=lambda x: x[1]['dominant_node_pct'], reverse=True)[:5]
    for site, stats in sorted_concentrated:
        print(f"  {site}: {stats['dominant_node_pct']:.1f}% in node {stats['dominant_node']} "
              f"({stats['total_samples']} samples, {stats['unique_nodes']} nodes)")
    
    # most diverse sites (high node diversity)
    print(f"\nMost Diverse Sites (High Node Diversity):")
    sorted_diverse = sorted(site_stats.items(), 
                          key=lambda x: x[1]['shannon_diversity'], reverse=True)[:5]
    for site, stats in sorted_diverse:
        print(f"  {site}: Diversity={stats['shannon_diversity']:.2f}, "
              f"{stats['unique_nodes']} nodes, {stats['total_samples']} samples, "
              f"dominant node={stats['dominant_node_pct']:.1f}%")
    
    # sites with widest spatial spread
    print(f"\nSites with Widest Node Distribution:")
    sorted_spread = sorted(site_stats.items(), 
                         key=lambda x: x[1]['spatial_spread'], reverse=True)[:5]
    for site, stats in sorted_spread:
        print(f"  {site}: Spread={stats['spatial_spread']:.2f}, "
              f"{stats['unique_nodes']} nodes across SOM")

print_node_diversity_stats(site_stats)

# create site-node distribution matrix
def create_site_node_matrix(som, site_node_mapping):
    """Create a matrix showing site distribution across nodes"""
    umatrix = som.distance_map()
    grid_shape = umatrix.shape
    
    # get all sites and nodes
    all_sites = sorted(site_node_mapping.keys())
    all_nodes = [(i, j) for i in range(grid_shape[0]) for j in range(grid_shape[1])]
    
    # create site presence matrix
    site_presence = {}
    for site in all_sites:
        presence_matrix = np.zeros(grid_shape)
        for (x, y) in site_node_mapping[site]:
            presence_matrix[x, y] += 1
        site_presence[site] = presence_matrix
    
    # plot site distribution heatmaps for top sites
    n_sites_to_plot = min(12, len(all_sites))
    top_sites = sorted(all_sites, key=lambda x: len(site_node_mapping[x]), reverse=True)[:n_sites_to_plot]
    
    n_cols = 4
    n_rows = (n_sites_to_plot + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, site in enumerate(top_sites):
        ax = axes[idx]
        im = ax.imshow(site_presence[site], cmap='Reds', origin='lower')
        ax.set_title(f'{site}\n({len(site_node_mapping[site])} samples)')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # hide empty subplots
    for idx in range(len(top_sites), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    for path in analysis_dirs:
        file_path = os.path.join(path, "site_node_distribution.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Site-node distribution saved: {file_path}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

create_site_node_matrix(som, site_node_mapping)

# save detailed results
node_diversity_df = pd.DataFrame.from_dict(site_stats, orient='index')
node_diversity_df.index.name = 'site'
node_diversity_df.reset_index(inplace=True)

for path in analysis_dirs:
    file_path = os.path.join(path, "node_diversity_per_site.csv")
    node_diversity_df.to_csv(file_path, index=False)
    print(f"Detailed node diversity analysis saved: {file_path}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("This analysis reveals:")
print("• CONCENTRATED SITES: Samples cluster in few nodes (environmental homogeneity)")
print("• DIVERSE SITES: Samples spread across many nodes (environmental variability)")
print("• SPATIAL SPREAD: How widely sites' samples distribute across the SOM")
print("• NODE COVERAGE: Efficiency of node usage per site")

# ----------------------------------------------------------------------
# Node diversity per site
# ----------------------------------------------------------------------
def create_site_node_mapping(bmu_sites):
    """Create reverse mapping: which sites go to which nodes"""
    site_to_nodes = defaultdict(list)
    for (x, y), sites in bmu_sites.items():
        for site in set(sites):  # Unique sites per node
            site_to_nodes[site].append((x, y))
    return site_to_nodes
    
def analyze_node_diversity_per_site(bmu_sites):
    """
    Computes how many distinct SOM nodes each site occupies (spatial dispersion).
    Sites with high node diversity are widespread; low = spatially cohesive.
    """
    site_node_counts = {site: len(set(nodes)) for site, nodes in create_site_node_mapping(bmu_sites).items()}
    total_nodes = sum(site_node_counts.values())

    site_diversity_df = (
        pd.DataFrame(list(site_node_counts.items()), columns=['Site', 'NodesOccupied'])
        .sort_values('NodesOccupied', ascending=False)
        .reset_index(drop=True)
    )

    # normalize if needed
    site_diversity_df['NodeShare_%'] = 100 * site_diversity_df['NodesOccupied'] / site_diversity_df['NodesOccupied'].sum()
    print("\nSITE NODE DIVERSITY SUMMARY:")
    print(site_diversity_df.head(10))

    return site_diversity_df


def plot_site_node_diversity(site_diversity_df, save=True):
    """Bar chart showing how many SOM nodes each site occupies."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=site_diversity_df, x='Site', y='NodesOccupied', palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of SOM Nodes Occupied")
    plt.title("Node Diversity per Site (Spatial Spread on SOM)")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save:
        for path in analysis_dirs:
            file_path = os.path.join(path, "site_node_diversity.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Site node diversity plot saved: {file_path}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

site_diversity_df = analyze_node_diversity_per_site(bmu_sites)
plot_site_node_diversity(site_diversity_df)

# ----------------------------------------------------------------------
# Node entropy
# ----------------------------------------------------------------------
from scipy.stats import entropy
def node_label_stats(bmu_locations, sample_labels):
    node_counts = {}
    for loc, lab in zip(map(tuple,bmu_locations), sample_labels):
        node_counts.setdefault(loc, Counter())[lab] += 1
    rows = []
    for node, cnt in node_counts.items():
        total = sum(cnt.values())
        freqs = np.array(list(cnt.values()))/total
        rows.append({'node':node, 'counts':dict(cnt), 'entropy': entropy(freqs)})
    return pd.DataFrame(rows)

node_stats = node_label_stats(bmu_locations, df2['site'].values)
node_stats.sort_values('entropy', ascending=False).head()

# ----------------------------------------------------------------------
# Analyze BMU level cluster vs SOM node level cluster alignment
# ----------------------------------------------------------------------
"""computes per-node statistics and global agreement metrics"""
def analyze_alignment(node_clusters, bmu_locations, cluster, som_shape):
    nx, ny = som_shape
    # containers
    node_sample_indices = {(i,j): [] for i in range(nx) for j in range(ny)}
    for idx, (x,y) in enumerate(bmu_locations):
        node_sample_indices[(x,y)].append(idx)

    # per-node metrics
    node_stats = []
    all_node_majority = np.full((nx, ny), -1, dtype=int)
    all_node_purity = np.zeros((nx, ny), dtype=float)
    all_node_entropy = np.zeros((nx, ny), dtype=float)
    all_node_disagreement_rate = np.zeros((nx, ny), dtype=float)

    def entropy_from_counts(counts):
        total = sum(counts)
        if total == 0: 
            return 0.0
        ps = np.array([c/total for c in counts if c>0])
        return -(ps * np.log2(ps)).sum()

    for i in range(nx):
        for j in range(ny):
            indices = node_sample_indices[(i,j)]
            if len(indices) == 0:
                node_stats.append({'node':(i,j),'cluster':0,'purity':np.nan,'entropy':np.nan,'majority_label':None,'disagreement_rate':np.nan})
                continue
            labels_here = [cluster[idx] for idx in indices]
            counts = Counter(labels_here)
            total = len(labels_here)
            majority_label, majority_count = counts.most_common(1)[0]
            purity = majority_count/total
            ent = entropy_from_counts(list(counts.values()))
            # disagreement = fraction of samples whose sample_label != node_clusters[i,j]
            node_cluster_label = int(node_clusters[i,j])
            disagree = sum(1 for lab in labels_here if lab != node_cluster_label)/total

            node_stats.append({'node':(i,j),'cluster':total,'purity':purity,'entropy':ent,'majority_label':majority_label,'disagreement_rate':disagree,'counts':counts})
            all_node_majority[i,j] = majority_label
            all_node_purity[i,j] = purity
            all_node_entropy[i,j] = ent
            all_node_disagreement_rate[i,j] = disagree

    # global agreement: compare sample-level labels against node-level label of its BMU
    sample_node_labels = np.array([ node_clusters[x,y] for (x,y) in bmu_locations ])
    cluster = np.array(cluster)
    # ARI/NMI between clusterings (two different clusterings of same samples)
    ari = adjusted_rand_score(cluster, sample_node_labels)
    nmi = normalized_mutual_info_score(cluster, sample_node_labels)

    return {
        'node_stats': pd.DataFrame(node_stats),
        'majority_matrix': all_node_majority,
        'purity_matrix': all_node_purity,
        'entropy_matrix': all_node_entropy,
        'disagreement_matrix': all_node_disagreement_rate,
        'global_ARI': ari,
        'global_NMI': nmi
    }
result = analyze_alignment(node_clusters=node_clusters, bmu_locations=bmu_locations, cluster=df2['cluster'], som_shape=som_shape)
result.keys()
result['node_stats'].head()
print(result['node_stats'].columns)
print(result['node_stats'].head())

"""
* Purity (majority fraction): 1.0 = all BMU samples in that neuron belong to the same BMU-cluster. Low purity (< ~0.6) → mixed signals / transitional node.

* Entropy: Shannon entropy of cluster labels in that node. Higher energy = more label mixing.

* Disagreement rate: fraction of BMU samples whose BMU-level label != neuron’s node-level label. High disagreement → misalignment.

* Global ARI / NMI: overall agreement between sample-level clusters and BMU→node cluster mapping. Values closer to 1 indicate strong agreement.

Thresholds:

1) Purity < 0.6 or disagreement_rate > 0.3 → mark node as transitional / misaligned.

2) Purity > 0.9 and disagreement_rate < 0.1 → stable, well-aligned.
"""

# plot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(result['purity_matrix'][::-1, :], cmap='YlGn', vmin=0, vmax=1, ax=axs[0])
plt.tight_layout()
axs[0].set_title('Node Purity (BMU Cluster Composition)')
sns.heatmap(result['disagreement_matrix'][::-1, :], cmap='Reds', vmin=0, vmax=1, ax=axs[1])
""" hot spots are misalignment """
axs[1].set_title('BMU–Node Cluster Disagreement Rate per node')
plt.tight_layout()
for path in analysis_dirs:
        try:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, "purity_matrix.png")
            plt.savefig(file_path, dpi=300)
            print(f"purity heatmap saved: {file_path}")
        except Exception as e:
            print(f"Error saving purity heatmaps to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()
            
# clustering consistency
print("Global Adjusted Rand Index (ARI):", result['global_ARI'])
print("Global Normalized Mutual Info (NMI):", result['global_NMI'])

# flag transitional nodes
df_nodes = result['node_stats']
transitional_nodes = df_nodes[
    (df_nodes['purity'] < 0.6) | (df_nodes['disagreement_rate'] > 0.3)
]
print(f"{len(transitional_nodes)} transitional nodes detected:")
print(transitional_nodes[['node', 'purity', 'disagreement_rate', 'majority_label']])

# plot mixed nodes
def plot_pies_on_nodes(node_stats, node_clusters, som_shape, n_clusters=n_clusters):
    """Plot small pies showing cluster composition per node with consistent colors"""
    # UNPACK the som_shape tuple
    if isinstance(som_shape, (int, float)):
        nx = ny = int(som_shape)  # Square grid
    else:
        nx, ny = som_shape  # Unpack tuple
    
    # consistent cluster coloring
    tab20 = plt.get_cmap('tab20')
    custom_indices = [0, 5, 10, 15, 19, 14]
    cluster_colors = [tab20(i) for i in custom_indices[:n_clusters]]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.invert_yaxis()  # Match SOM orientation
    ax.set_aspect('equal')
    ax.set_title("Node Composition (BMU-level Cluster Mix)")

    # check what's in node_stats
    print(f"Type of node_stats: {type(node_stats)}")
    if hasattr(node_stats, 'keys'):
        print(f"node_stats keys: {node_stats.keys()}")
    
    # Handle different possible data structures
    if isinstance(node_stats, dict):
        # If node_stats is a dictionary with (x,y) keys
        for (i, j), stats in node_stats.items():
            # Check if stats has cluster_counts or if it's the counts directly
            if isinstance(stats, dict) and 'cluster_counts' in stats:
                counts = stats['cluster_counts']
            elif isinstance(stats, dict):
                counts = stats  # stats is the counts dictionary itself
            else:
                print(f"Skipping node ({i},{j}): unexpected stats type {type(stats)}")
                continue
                
            if not counts or not isinstance(counts, dict):
                continue
                
            # prepare data for pie chart
            labels = list(counts.keys())
            sizes = list(counts.values())
            
            # map cluster numbers to colors
            pie_colors = [cluster_colors[cluster_id] for cluster_id in labels]
            
            # pie chart
            pie_radius = 0.35
            wedges, texts = ax.pie(sizes, 
                                  colors=pie_colors,
                                  radius=pie_radius,
                                  center=(i, j))  # Center at node position
            
            # adjust transparency based on sample count (optional)
            total_samples = sum(sizes)
            alpha = min(0.3 + (total_samples / 50), 1.0)  # Scale alpha with sample count
            for wedge in wedges:
                wedge.set_alpha(alpha)
                
    elif isinstance(node_stats, pd.DataFrame):
        # If node_stats is a DataFrame
        for _, row in node_stats.iterrows():
            # Handle different possible column names
            if 'node' in row:
                i, j = row['node']
            elif 'x' in row and 'y' in row:
                i, j = row['x'], row['y']
            else:
                continue
                
            # Get counts from different possible column names
            counts = None
            for col in ['counts', 'cluster_counts', 'cluster_composition']:
                if col in row and isinstance(row[col], dict):
                    counts = row[col]
                    break
            
            if not counts or not isinstance(counts, dict):
                continue
                
            # Prepare data for pie chart
            labels = list(counts.keys())
            sizes = list(counts.values())
            
            # Map cluster numbers to colors
            pie_colors = [cluster_colors[cluster_id] for cluster_id in labels]
            
            # Create small pie chart
            pie_radius = 0.35
            wedges, texts = ax.pie(sizes, 
                                  colors=pie_colors,
                                  radius=pie_radius,
                                  center=(i, j))
            
            # Optional: Adjust transparency based on sample count
            total_samples = sum(sizes)
            alpha = min(0.3 + (total_samples / 50), 1.0)
            for wedge in wedges:
                wedge.set_alpha(alpha)
    else:
        print(f"Unsupported node_stats type: {type(node_stats)}")
        return

    # ~ # node borders
    # ~ for i in range(nx):
        # ~ for j in range(ny):
            # ~ rect = plt.Rectangle((i-0.5, j-0.5), 1, 1, 
                               # ~ fill=False, edgecolor='gray', lw=0.5)
            # ~ ax.add_patch(rect)
    
    # legend with the same clustering colors
    legend_elements = [
        mpatches.Patch(color=cluster_colors[i], label=f'Cluster {i}')
        for i in range(n_clusters)
    ]
    ax.legend(handles=legend_elements, title="KMeans Clusters",
             bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # save to result directories
    for path in analysis_dirs:
        file_path = os.path.join(path, "node_composition_pies.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Node composition pies saved: {file_path}")
    
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
plot_pies_on_nodes(result['node_stats'], node_clusters, som_shape)

# overlay misaligned nodes on umatrix
umatrix = som.distance_map() 
u = umatrix
plt.imshow(u, cmap='bone_r')
for (i, j) in transitional_nodes['node']:
    plt.gca().add_patch(plt.Circle((i, j), 0.3, color='red', fill=True, alpha=0.6))
plt.title('U-Matrix with Transitional (Misaligned) Nodes Highlighted')
plt.tight_layout()
for path in analysis_dirs:
        try:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, "misaligned_nodes_overlay.png")
            plt.savefig(file_path, dpi=300)
            print(f"Plot saved: {file_path}")
        except Exception as e:
            print(f"Error saving plot to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

# summary
summary = result['node_stats'][['purity', 'entropy', 'disagreement_rate']].describe()
print(summary)

"""Node-level purity ranged from 0.42–1.0 (mean = 0.78), while disagreement rates averaged 0.18 ± 0.12, suggesting moderate structural consistency between BMU- and node-level cluster representations."""

# ----------------------------------------------------------------------
# Within-node homogeneity - how similar the sites mapped to the same SOM node
# ----------------------------------------------------------------------
def compute_within_node_distances(df, som, X_scaled, metrics=["cosine"]):
    """Compute within-node distances"""
    node_samples = defaultdict(list)
    
    for idx, x in enumerate(X_scaled):
        node = som.winner(x)
        node_samples[node].append(idx)
    
    results = []
    for node, idxs in node_samples.items():
        if len(idxs) < 2:
            continue
        data = X_scaled[idxs]
        
        for metric in metrics:
            d = pdist(data, metric=metric)
            results.append({
                "node": node,
                "metric": metric,
                "mean_distance": np.mean(d),
                "std_distance": np.std(d),
                "n_samples": len(idxs)
            })
    return pd.DataFrame(results)

within_df = compute_within_node_distances(df2, som, X_scaled)

# plot
plt.figure(figsize=(10, 6))
sns.barplot(data=within_df, x="metric", y="mean_distance", hue="node", ci=None)
plt.ylabel("Mean Within-node Distance")
plt.title("Within-node Homogeneity by Metric")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
for path in analysis_dirs:
        try:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, "within_node_Homogeneity.png")
            plt.savefig(file_path, dpi=300)
            print(f"Plot saved: {file_path}")
        except Exception as e:
            print(f"Error saving plot to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

'''
Each node has an average intra-node distance per metric.
Lower mean distance = more homogeneous sites within that node.
Compare across metrics:
    cosine: absolute spread.
    cosine: angular spread (direction similarity).
    Mahalanobis: adjusted for correlations in variables.
'''
# ----------------------------------------------------------------------
# Between-node distances (SOM weights centroids) - how far apart the SOM node centroids (weights) are in feature space
# ----------------------------------------------------------------------
def compute_between_node_distances(weights, metrics=["cosine"]):
    results = {}
    for metric in metrics:
        D = cdist(weights, weights, metric=metric)
        results[metric] = D
    return results

# flatten SOM weights to (#nodes, n_features)
weights = som.get_weights().reshape(-1, som.get_weights().shape[-1])

# compute distances
between_dists = compute_between_node_distances(weights)

# --- Visualization: heatmaps ---
fig, axes = plt.subplots(1, len(between_dists), figsize=(5*len(between_dists), 5))
if len(between_dists) == 1:
    axes = [axes]  # keep iterable

for ax, (metric, D) in zip(axes, between_dists.items()):
    sns.heatmap(D, cmap="viridis", ax=ax, square=True, cbar=True)
    ax.set_title(f"Between-node {metric.capitalize()} Distances")

plt.tight_layout()
for path in analysis_dirs:
        try:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, "between_node_distance.png")
            plt.savefig(file_path, dpi=300)
            print(f"Plot saved: {file_path}")
        except Exception as e:
            print(f"Error saving plot to {path}: {e}")
plt.show(block=False)
plt.pause(1)
plt.close()

'''
Heatmaps show how far node centroids are in feature space.
Dark vs bright blocks → cluster separation.
Mahalanobis will “stretch/shrink” along correlated axes (ecologically meaningful if chemical/environmental drivers co-vary).
'''

# ----------------------------------------------------------------------
# U-Matrix + selected driver overlays
# ----------------------------------------------------------------------
'''
def plot_category_overlays(som, df, category_list, X_scaled):
    fig, axes = plt.subplots(1, len(category_list) + 1, figsize=(5 * (len(category_list) + 1), 5))

    # Panel A: U-Matrix
    axes[0].pcolor(som.distance_map().T, cmap='bone_r')
    axes[0].set_title('U-Matrix')

    for i, category in enumerate(category_list):
        category_density = np.zeros((som._weights.shape[0], som._weights.shape[1]))

        for idx, row in df.iterrows():
            x, y = som.winner(X_scaled[idx])
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
def plot_category_overlays(som, df, category_list, X_scaled, n_cols=4):
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
            x, y = som.winner(X_scaled[idx])
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

def plot_category_overlays(som, df, category_list, X_scaled, n_cols=4):
    n_categories = len(category_list)
    n_panels = n_categories + 1  # +1 for U-Matrix
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Panel A: U-Matrix
    u_matrix = som.distance_map()
    im0 = axes[0].imshow(u_matrix, cmap='bone_r', origin='lower')
    axes[0].set_title('cosine-based U-Matrix', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    for i, category in enumerate(category_list):
        category_density = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1]))

        for idx, row in df.iterrows():
            x, y = som.winner(X_scaled[idx])
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

plot_category_overlays(som, df2, env_vars, X_scaled, n_cols=4)

# ----------------------------------------------------------------------
# Compute BMU-wise averages
# ----------------------------------------------------------------------
'''def compute_bmu_averages(df, X_scaled, som, species_vars, chemical_vars):
    n_nodes_x, n_nodes_y = som._weights.shape[:2]
    bmu_species = np.zeros((n_nodes_x, n_nodes_y, len(species_vars)))
    bmu_chemicals = np.zeros((n_nodes_x, n_nodes_y, len(chemical_vars)))
    bmu_counts = np.zeros((n_nodes_x, n_nodes_y))

    for idx, row in df.iterrows():
        x, y = som.winner(X_scaled[idx])
        bmu_species[x, y] += row[species_vars].fillna(0).values
        bmu_chemicals[x, y] += row[chemical_vars].fillna(0).values
        bmu_counts[x, y] += 1

    # Average over counts
    bmu_species_avg = np.divide(bmu_species, bmu_counts[..., np.newaxis], out=np.zeros_like(bmu_species), where=bmu_counts[..., np.newaxis]!=0)
    bmu_chemicals_avg = np.divide(bmu_chemicals, bmu_counts[..., np.newaxis], out=np.zeros_like(bmu_chemicals), where=bmu_counts[..., np.newaxis]!=0)

    return bmu_species_avg, bmu_chemicals_avg

bmu_species_avg, bmu_chemicals_avg = compute_bmu_averages(df1, X_scaled, som, major_cats, env_vars)'''


def compute_bmu_averages(df, X_scaled, som, sp_vars, env_vars):
    n_nodes_x, n_nodes_y = som.get_weights().shape[:2]
    bmu_species = np.zeros((n_nodes_x, n_nodes_y, len(sp_vars)))
    bmu_envs = np.zeros((n_nodes_x, n_nodes_y, len(env_vars)))
    bmu_counts = np.zeros((n_nodes_x, n_nodes_y))

    for idx in range(len(df)):
        x, y = som.winner(X_scaled[idx])
        bmu_species[x, y] += df.loc[idx, sp_vars].fillna(0).values
        bmu_envs[x, y] += df.loc[idx, env_vars].fillna(0).values
        bmu_counts[x, y] += 1

    # Avoid division by zero
    bmu_species_avg = np.divide(
        bmu_species, bmu_counts[..., np.newaxis], 
        out=np.zeros_like(bmu_species), 
        where=bmu_counts[..., np.newaxis] != 0
    )
    bmu_envs_avg = np.divide(
        bmu_envs, bmu_counts[..., np.newaxis], 
        out=np.zeros_like(bmu_envs), 
        where=bmu_counts[..., np.newaxis] != 0
    )

    return bmu_species_avg, bmu_envs_avg

# ~ bmu_sp_avg, bmu_env_avg = compute_bmu_averages(df2, X_scaled, som, sp_vars, env_vars)

# ----------------------------------------------------------------------
# Correlation analysis between species density and env profiles
# ----------------------------------------------------------------------
def plot_species_environment_correlation(bmu_species_avg, bmu_env_avg, species_list, env_vars, threshold=0.5):
    # flatten and clean BMU data
    bmu_species_flat = bmu_species_avg.reshape(-1, bmu_species_avg.shape[-1])
    bmu_env_flat = bmu_env_avg.reshape(-1, bmu_env_avg.shape[-1])

    species_df = pd.DataFrame(bmu_species_flat, columns=species_list)
    env_df = pd.DataFrame(bmu_env_flat, columns=env_vars)

    valid_mask = ~(species_df.isna() | (species_df == 0)).all(axis=1) & ~(env_df.isna() | (env_df == 0)).all(axis=1)
    species_df = species_df[valid_mask]
    env_df = env_df[valid_mask]

    # apply major category aggregation if labels_dict is provided
    if labels_dict is not None:
        # map species to major categories
        species_to_category = {}
        for species in species_list:
            if species in labels_dict:
                species_to_category[species] = labels_dict[species]
            else:
                species_to_category[species] = 'Other'  # Default category
        
        # aggregate species by major categories
        category_aggregated = {}
        for category in set(species_to_category.values()):
            category_species = [sp for sp in species_list if species_to_category[sp] == category]
            if category_species:
                category_aggregated[category] = species_df[category_species].mean(axis=1)
        
        # create new aggregated dataframe
        aggregated_df = pd.DataFrame(category_aggregated)
        species_list_agg = list(category_aggregated.keys())
        species_df_used = aggregated_df
    else:
        # use original species data
        species_list_agg = species_list
        species_df_used = species_df

    # calculate correlations
    correlations = pd.DataFrame(index=species_list_agg, columns=env_vars)
    annotations = pd.DataFrame(index=species_list_agg, columns=env_vars)

    for sp in species_list_agg:
        for env in env_vars:
            if species_df_used[sp].isnull().all() or env_df[env].isnull().all():
                correlations.loc[sp, env] = np.nan
                annotations.loc[sp, env] = ""
            else:
                corr = species_df_used[sp].corr(env_df[env], method="pearson")
                correlations.loc[sp, env] = corr
                # annotate strong correlations
                annotations.loc[sp, env] = f"{corr:.2f}" if abs(corr) >= threshold else ""

    correlations = correlations.astype(float)

    # visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # plot 1: species-level correlations (if not aggregated)
    if labels_dict is None:
        sns.heatmap(correlations, annot=annotations, fmt="", cmap='RdBu_r', center=0, 
                   linewidths=0.5, linecolor='gray', ax=ax1)
        ax1.set_title(f'Species–Environment Correlation (|r| ≥ {threshold})')
    else:
        # plot aggregated correlations
        sns.heatmap(correlations, annot=annotations, fmt="", cmap='RdBu_r', center=0, 
                   linewidths=0.5, linecolor='gray', ax=ax1)
        ax1.set_title(f'Major Category–Environment Correlation (|r| ≥ {threshold})')

    # plot 2: Strongest correlations bar plot
    strong_correlations = []
    for sp in species_list_agg:
        for env in env_vars:
            corr_val = correlations.loc[sp, env]
            if abs(corr_val) >= threshold and not np.isnan(corr_val):
                strong_correlations.append({
                    'Category': sp,
                    'Environment': env,
                    'Correlation': corr_val,
                    'Abs_Correlation': abs(corr_val)
                })
    
    if strong_correlations:
        strong_df = pd.DataFrame(strong_correlations)
        strong_df = strong_df.nlargest(15, 'Abs_Correlation')  # Top 15 strongest
        
        colors = ['red' if x < 0 else 'blue' for x in strong_df['Correlation']]
        y_pos = np.arange(len(strong_df))
        
        ax2.barh(y_pos, strong_df['Correlation'], color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"{row['Category']}-{row['Environment']}" 
                           for _, row in strong_df.iterrows()])
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_title('Strongest Species-Environment Correlations')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # add correlation values
        for i, (_, row) in enumerate(strong_df.iterrows()):
            ax2.text(row['Correlation'] + (0.02 if row['Correlation'] >= 0 else -0.05), 
                    i, f"{row['Correlation']:.3f}", 
                    ha='left' if row['Correlation'] >= 0 else 'right', 
                    va='center', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, f'No correlations ≥ |{threshold}|', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Strong Correlations (None found)')

    plt.tight_layout()

    for path in result_dirs:
        try:
            os.makedirs(path, exist_ok=True)
            suffix = "aggregated" if labels_dict else "detailed"
            file_path = os.path.join(path, f"species_environment_correlation_{suffix}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            correlations.to_csv(os.path.join(path, f"correlation_matrix_{suffix}.csv"))
            
            print(f"✓ Correlation analysis saved: {file_path}")
        except Exception as e:
            print(f"✗ Error saving to {path}: {e}")

    plt.show()
    
    return correlations

# ~ plot_species_environment_correlation(bmu_sp_avg, bmu_env_avg, sp_vars, env_vars, threshold=0.5)

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
        for path in result_dirs:
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

signature_results = identify_signature_species_per_cluster(
    df=df2,
    cluster_col='SOM_Cluster',  # or 'cluster' for K-means
    species_vars=sp_vars,       # your species variables
    labels_dict=labels_dict,    # your category mapping
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
colors = plt.cm.tab20.colors[:n_clusters]
label_to_index = {label: i for i, label in enumerate(unique_labels)}

# calculate cluster centers in SOM grid space:
'''draw arrows between clusters or place labels'''
labels = df2['SOM_Cluster'].values
unique_labels = np.unique(labels)
n_clusters = len(unique_labels)
cluster_centers = {}
for cluster_id in range(n_clusters):
    coords = np.argwhere(node_clusters == cluster_id)
    if len(coords) > 0:
        center = coords.mean(axis=0)
        cluster_centers[cluster_id] = center

# calculate cluster vectors in feature space:
'''averages SOM weight vectors of each node in the cluster: to compare clusters'''
cluster_vectors = {}
for cluster_id in range(n_clusters):
    indices = np.argwhere(node_clusters == cluster_id).reshape(-1, 2)
    if len(indices) > 0:
        node_vectors = [som.get_weights()[x, y] for x, y in indices]
        cluster_vectors[cluster_id] = np.mean(node_vectors, axis=0)

# ----------------------------------------------------------------------
# Transition zone analysis
# ----------------------------------------------------------------------
def find_transition_pairs(node_clusters):
    """detect all unique adjacent cluster pairs in the SOM"""
    transitions = set()
    for i in range(node_clusters.shape[0]):
        for j in range(node_clusters.shape[1]):
            current = node_clusters[i, j]
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:  # check 4-neighborhood
                ni, nj = i + dx, j + dy
                if 0 <= ni < node_clusters.shape[0] and 0 <= nj < node_clusters.shape[1]:
                    neighbor = node_clusters[ni, nj]
                    if neighbor != current:
                        pair = tuple(sorted((current, neighbor)))
                        transitions.add(pair)
    return sorted(transitions)

def get_neighboring_clusters(node, node_clusters):
    """get set of neighboring cluster IDs for a node: detects transition zones """
    x, y = node
    neighbors = set()
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < node_clusters.shape[0] and 0 <= ny < node_clusters.shape[1]:
                neighbors.add(node_clusters[nx, ny])
    return neighbors

# ~ '''    

# Identify transition(edge) components between two clusters only
def plot_transition_zone(cluster_A, cluster_B, node_clusters, bmu_locations, 
                        umatrix, xx, yy, xrange, yrange, X_scaled, df2, 
                        variables_of_interest, som_shape, figsize=(14, 12)):
    """transition zone visualization consistent with kmeans (tab20) cluster coloring"""

    # kmeans cluster assignment:
    tab20 = plt.get_cmap('tab20')
    custom_indices = [0, 5, 10, 15, 19, 14]
    discrete_tab20 = ListedColormap([tab20(i) for i in custom_indices])
    cluster_ids = np.unique(node_clusters)
    cluster_colors = {cluster: tab20(custom_indices[cluster]) for cluster in cluster_ids}
    cluster_alphas = {cluster: 0.25 if cluster in [cluster_A, cluster_B] else 0.1 for cluster in cluster_ids}

    # 1. identify boundaries
    transitional_samples = []
    transitional_weights = []
    transition_nodes = set()
    
    for idx, (x, y) in enumerate(bmu_locations):
        neighbors = get_neighboring_clusters((x, y), node_clusters)
        if cluster_A in neighbors and cluster_B in neighbors:
            transitional_samples.append(idx)
            transitional_weights.append(X_scaled[idx])
            transition_nodes.add((x, y))
    
    if not transitional_samples:
        print(f"No transition samples between Cluster {cluster_A} and {cluster_B}")
        return None
    
    # 2. create figure
    plt.switch_backend('TkAgg')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
    
    # 3. left subplot: SOM visualization
    norm_umatrix = (umatrix - umatrix.min()) / (umatrix.max() - umatrix.min())
    
    # plot rectangular U-Matrix background
    ax1.imshow(umatrix, cmap='bone_r', alpha=0.7,
              extent=[xrange[0]-0.5, xrange[-1]+0.5, 
                     yrange[-1]+0.5, yrange[0]-0.5],
              aspect='auto')
    
    # highlight all clusters
    for cluster in cluster_ids:
        color = discrete_tab20[cluster]
        alpha = cluster_alphas[cluster]
        for (x, y) in np.argwhere(node_clusters == cluster):
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                facecolor=color,
                                alpha=alpha,
                                edgecolor='none')
            ax1.add_patch(rect)
    
    # highlight focus clusters more prominently
    for cluster in [cluster_A, cluster_B]:
        color = discrete_tab20[cluster]
        for (x, y) in np.argwhere(node_clusters == cluster):
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                facecolor=color, alpha=0.6,
                                edgecolor='black', linewidth=1)
            ax1.add_patch(rect)
            
    # highlight transition nodes
    for cluster in [cluster_A, cluster_B]:
        color = discrete_tab20[cluster]
        for (x, y) in np.argwhere(node_clusters == cluster):
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                facecolor=color, alpha=0.6,
                                edgecolor='black', linewidth=1)
            ax1.add_patch(rect)
    
    # calculate centroids
    def get_centroid(cluster_id):
        coords = np.argwhere(node_clusters == cluster_id)
        if len(coords) == 0:
            return (0, 0)
        x_mean = np.mean([x for x, y in coords])
        y_mean = np.mean([y for x, y in coords])
        return (x_mean, y_mean)
    
    centroid_A = get_centroid(cluster_A)
    centroid_B = get_centroid(cluster_B)
    
    # plot centroids with matching cluster colors
    color_A = discrete_tab20[cluster_A]
    color_B = discrete_tab20[cluster_B]
    ax1.plot(*centroid_A, 'o', color=color_A, markersize=18, 
            markeredgecolor='white', markeredgewidth=1.5)
    ax1.plot(*centroid_B, 'o', color=color_B, markersize=18,
            markeredgecolor='white', markeredgewidth=1.5)
    
    # add transition arrow
    ax1.annotate('', xy=centroid_B, xytext=centroid_A,
                arrowprops=dict(arrowstyle='->', color='black', 
                              linewidth=2, linestyle='-', alpha=0.7))
    
    # legend
    legend_elements = [
        Patch(facecolor=discrete_tab20[cluster_A], alpha=0.6, label=f'Cluster {cluster_A}'),
        Patch(facecolor=discrete_tab20[cluster_B], alpha=0.6, label=f'Cluster {cluster_B}'),
        # ~ Patch(facecolor='red', alpha=0.5, label='Transition Nodes'),
        # ~ *[Patch(facecolor=cluster_colors[i], alpha=cluster_alphas[i], label=f'Cluster {i}') 
            # ~ for i in cluster_ids if i not in [cluster_A, cluster_B]][:4]
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    ax1.set_title(f'Transition Zone: Cluster {cluster_A} → {cluster_B}\n'
                f'{len(transitional_samples)} Transition Samples', pad=15)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
    
    # --- right subplot: Feature Differences
    transition_df = df2.iloc[transitional_samples] # extract feature values
    feature_means = transition_df[variables_of_interest].mean().sort_values(ascending=False) # mean of each feature 
    global_means = df2[variables_of_interest].mean()
    feature_diff = ((feature_means - global_means[feature_means.index]) / global_means[feature_means.index]).sort_values(ascending=False)

    cmap = plt.get_cmap('coolwarm')
    colors = [cmap(0.5 + 0.5 * (val / max(abs(feature_diff)))) for val in feature_diff.head(15)]

    bars = ax2.barh(feature_diff.head(15).index, feature_diff.head(15), color=colors, edgecolor='black')
    for bar in bars:
        width = bar.get_width()
        ax2.text(width/2, bar.get_y() + bar.get_height()/2, f'{width:.1%}', ha='center', va='center', color='white', fontweight='bold')

    ax2.set_xlabel('Relative Difference from Global Mean')
    ax2.set_title(f'Top 15 Discriminating Features\nCluster {cluster_A} → Cluster {cluster_B}')
    ax2.axvline(0, color='gray', linestyle='--')

    plt.tight_layout()
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

'''
# Plot transition zones for different cluster pairs
plot_transition_zone(cluster_A=0, cluster_B=1, node_clusters=node_clusters, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_scaled=X_scaled,
                    df2=df2, variables_of_interest=variables_of_interest, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=0, cluster_B=2, node_clusters=node_clusters, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_scaled=X_scaled, 
                    df2=df2, variables_of_interest=variables_of_interest, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=0, cluster_B=3, node_clusters=node_clusters, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_scaled=X_scaled, 
                    df2=df2, variables_of_interest=variables_of_interest, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=0, cluster_B=4, node_clusters=node_clusters, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_scaled=X_scaled, 
                    df2=df2, variables_of_interest=variables_of_interest, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=1, cluster_B=2, node_clusters=node_clusters, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_scaled=X_scaled, 
                    df2=df2, variables_of_interest=variables_of_interest, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=1, cluster_B=3, node_clusters=node_clusters, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_scaled=X_scaled, 
                    df2=df2, variables_of_interest=variables_of_interest, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=1, cluster_B=4, node_clusters=node_clusters, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_scaled=X_scaled, 
                    df2=df2, variables_of_interest=variables_of_interest, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=2, cluster_B=3, node_clusters=node_clusters, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_scaled=X_scaled, 
                    df2=df2, variables_of_interest=variables_of_interest, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=2, cluster_B=4, node_clusters=node_clusters, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_scaled=X_scaled, 
                    df2=df2, variables_of_interest=variables_of_interest, 
                    som_shape=som_shape)

plot_transition_zone(cluster_A=3, cluster_B=4, node_clusters=node_clusters, 
                    bmu_locations=bmu_locations, umatrix=umatrix, xx=xx, yy=yy, 
                    xrange=xrange, yrange=yrange, X_scaled=X_scaled, 
                    df2=df2, variables_of_interest=variables_of_interest, 
                    som_shape=som_shape)
'''
                    
# ----------------------------------------------------------------------
# Identify transition samples between all clusters
# ----------------------------------------------------------------------

# kmeans cluster color assignment:
tab20 = plt.get_cmap('tab20')
custom_indices = [0, 5, 10, 15, 19, 14]
discrete_color = [tab20(i) for i in custom_indices]
discrete_tab20 = ListedColormap(discrete_color) # usage: discrete_tab20() i.e. cluster_color(i)/(c1)/(c2)

cluster_ids = np.unique(node_clusters)
cluster_colors = {cluster: tab20(custom_indices[cluster]) for cluster in cluster_ids} # usage: cluster_colors[] i.e. cluster_color[i]/[c1]/[c2]

# create transition zone plot
plt.figure(figsize=(14, 12))
ax = plt.gca()
ax.set_aspect('equal')

# background U-Matrix
plt.imshow(umatrix, cmap='bone_r', origin='lower', alpha=0.7)

# plot cluster boundaries
for i in range(node_clusters.shape[0]):
    for j in range(node_clusters.shape[1]):
        current = node_clusters[i, j]
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + dx, j + dy
            if 0 <= ni < node_clusters.shape[0] and 0 <= nj < node_clusters.shape[1]:
                neighbor = node_clusters[ni, nj]
                if neighbor != current:
                    plt.plot([i, ni], [j, nj], 'k-', linewidth=1.5, alpha=0.5)

# transition markers
transition_pairs = find_transition_pairs(node_clusters)
edge_colors = plt.cm.tab10(np.linspace(0, 1, len(transition_pairs)))

for pair_idx, (c1, c2) in enumerate(transition_pairs):
    indices = []
    for idx, (x, y) in enumerate(bmu_locations):
        neighbors = get_neighboring_clusters((x, y), node_clusters)
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
pca_df['Cluster'] = node_clusters.ravel()

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
