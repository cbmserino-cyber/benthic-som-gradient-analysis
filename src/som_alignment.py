## =====================================================================
## Objective: Analyze trained SOM using benthic cover (not per site) to uncover gradients in composition space and link them to environmental gradient
## Output: purity/entropy/disagreement + ARI/NMI 
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
# Analyze node diversity per site
# ----------------------------------------------------------------------
def analyze_node_diversity_per_site(som, df, site_col='site'):
    """
    analyze how each site's samples are distributed across SOM nodes
    ** measures of node diversity for each site
    *** how spread sites are distributed across SOM grid/topology
    """
    from collections import defaultdict
    
    # BMU for each sample and group by site
    site_node_mapping = defaultdict(list)
    for idx, row in df.iterrows():
        x, y = som.winner(X_hel[idx])
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


# First call the function to get site_stats
site_stats, site_node_mapping = analyze_node_diversity_per_site(som, df2, site_col='site')


# notable sites
def prepare_plot_df(site_stats):
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

    notable_sites = (
        set(plot_df.nlargest(3, 'shannon_diversity')['site']) |
        set(plot_df.nsmallest(3, 'shannon_diversity')['site']) |
        set(plot_df.nlargest(3, 'total_samples')['site'])
    )

    return plot_df, notable_sites


# Now prepare the plot data
plot_df, notable_sites = prepare_plot_df(site_stats)


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

    
    # identify notable sites
    top_diversity = plot_df.nlargest(3, 'shannon_diversity')['site']
    low_diversity = plot_df.nsmallest(3, 'shannon_diversity')['site']
    large_samples = plot_df.nlargest(3, 'total_samples')['site']
    
    # subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # (1) node diversity vs sample size
    scatter = axes[0, 0].scatter(plot_df['total_samples'], plot_df['shannon_diversity'],
                                c=plot_df['unique_nodes'], cmap='viridis', s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Total Samples')
    axes[0, 0].set_ylabel('Node Diversity (Shannon)')
    axes[0, 0].set_title('Node Diversity vs Sample Size')
    plt.colorbar(scatter, ax=axes[0, 0], label='Unique Nodes')
    '''
    # add all site labels
    for idx, row in plot_df.iterrows():
        axes[0, 0].annotate(row['site'], (row['total_samples'], row['shannon_diversity']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)'''
    # add only notable sites                       
    for _, row in plot_df.iterrows():
        if row['site'] in notable_sites:
            axes[0, 0].annotate(
                row['site'],
                (row['total_samples'], row['shannon_diversity']),
                xytext=(6, 6),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold'
            )                       
    
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


# Now call the plotting function
plot_node_diversity_per_site(site_stats)


# Individual plots

# (1) Node diversity vs sample size
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(
    plot_df['total_samples'],
    plot_df['shannon_diversity'],
    c=plot_df['unique_nodes'],
    cmap='viridis',
    s=100,
    alpha=0.7
)

ax.set_xlabel('Total Samples')
ax.set_ylabel('Node Diversity (Shannon)')
ax.set_title('Node Diversity vs Sample Size')
plt.colorbar(sc, ax=ax, label='Unique Nodes')

for _, row in plot_df.iterrows():
    if row['site'] in notable_sites:
        ax.annotate(
            row['site'],
            (row['total_samples'], row['shannon_diversity']),
            xytext=(6, 6),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold'
        )
plt.tight_layout()
for path in analysis_dirs:
    file_path = os.path.join(path, "node_diversity_per_site.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Node diversity per site saved: {file_path}")
plt.show(block=False)
plt.pause(1)
plt.close()

# (2) Dominant node percentage
N_TOP_SITES = 15
top_sites_df = plot_df.nlargest(N_TOP_SITES, 'dominant_node_pct')
fig, ax = plt.subplots(figsize=(10, 6))
# Sort for plotting
top_sites_sorted = top_sites_df.sort_values('dominant_node_pct')
bars = ax.barh(range(len(top_sites_sorted)), top_sites_sorted['dominant_node_pct'])
# Color gradient from red (high concentration) to orange
colors = plt.cm.Reds(np.linspace(0.6, 0.9, len(top_sites_sorted)))
for i, bar in enumerate(bars):
    bar.set_color(colors[i])

# Add value labels
for i, (_, row) in enumerate(top_sites_sorted.iterrows()):
    ax.text(row['dominant_node_pct'] + 1, i, 
            f"{row['dominant_node_pct']:.1f}%", 
            va='center', fontsize=9)

ax.set_yticks(range(len(top_sites_sorted)))
ax.set_yticklabels(top_sites_sorted['site'], fontsize=10)
ax.set_xlabel('Dominant Node Percentage (%)', fontsize=12)
ax.set_title(f'Top {N_TOP_SITES} Most Concentrated Sites\n(% of samples in single SOM node)', 
             fontweight='bold', fontsize=14)
ax.set_xlim([0, 105])  # Allow space for labels
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()

for path in analysis_dirs:
    try:
        file_path = os.path.join(path, f"top_{N_TOP_SITES}_concentrated_sites.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"✓ Top {N_TOP_SITES} concentrated sites plot saved: {file_path}")
    except Exception as e:
        print(f"✗ Error saving to {path}: {e}")

plt.show(block=False)
plt.pause(1)
plt.close()

# (3) Unique nodes per site
fig, ax = plt.subplots(figsize=(7, 6))

df = plot_df.sort_values('unique_nodes')
ax.barh(df['site'], df['unique_nodes'])

ax.set_xlabel('Number of Unique Nodes')
ax.set_title('Node Richness per Site')

plt.tight_layout()
for path in analysis_dirs:
    file_path = os.path.join(path, "node_uniques_persite.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Node diversity per site saved: {file_path}")
plt.show(block=False)
plt.pause(1)
plt.close()

# (4) Node coverage
fig, ax = plt.subplots(figsize=(7, 6))

df = plot_df.sort_values('node_coverage')
ax.barh(df['site'], df['node_coverage'])

ax.set_xlabel('Node Coverage (%)')
ax.set_title('Node Coverage per Site')

plt.tight_layout()
for path in analysis_dirs:
    file_path = os.path.join(path, "node_coverage.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Node diversity per site saved: {file_path}")
plt.show(block=False)
plt.pause(1)
plt.close()

# (5) Spatial spread
# Get top sites for spatial spread analysis
N_TOP_SITES = 15

# Top spatially spread sites (highest spatial spread)
top_spread_sites = plot_df.nlargest(N_TOP_SITES, 'spatial_spread')['site'].tolist()
top_spread_df = plot_df[plot_df['site'].isin(top_spread_sites)].copy()

# Top clustered sites (lowest spatial spread)
top_clustered_sites = plot_df.nsmallest(N_TOP_SITES, 'spatial_spread')['site'].tolist()

# ------------------------------------------------------------
# PLOT 1: Top N spatially spread sites (bar chart)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

# Sort for plotting
top_spread_sorted = top_spread_df.sort_values('spatial_spread', ascending=True)

# Create bars with gradient color
bars = ax.barh(range(len(top_spread_sorted)), top_spread_sorted['spatial_spread'])

# Color gradient from light blue to dark blue
colors = plt.cm.Blues(np.linspace(0.6, 0.9, len(top_spread_sorted)))
for i, bar in enumerate(bars):
    bar.set_color(colors[i])

# Add value labels
for i, (_, row) in enumerate(top_spread_sorted.iterrows()):
    ax.text(row['spatial_spread'] + 0.02, i, 
            f"{row['spatial_spread']:.2f}", 
            va='center', fontsize=9, fontweight='bold')

ax.set_yticks(range(len(top_spread_sorted)))
ax.set_yticklabels(top_spread_sorted['site'], fontsize=10)
ax.set_xlabel('Spatial Spread Index', fontsize=12)
ax.set_title(f'Top {N_TOP_SITES} Spatially Spread Sites\n(Highest Dispersion Across SOM Grid)', 
             fontweight='bold', fontsize=14)
ax.set_xlim([0, top_spread_sorted['spatial_spread'].max() * 1.1])  # Add padding for labels
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()

# Save Plot 1
for path in analysis_dirs:
    try:
        file_path = os.path.join(path, f"top_{N_TOP_SITES}_spatially_spread_sites.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"✓ Top {N_TOP_SITES} spatially spread sites plot saved: {file_path}")
    except Exception as e:
        print(f"✗ Error saving to {path}: {e}")

plt.show(block=False)
plt.pause(1)
plt.close()

# ------------------------------------------------------------
# PLOT 2: Spread vs Concentration scatter plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

# Create scatter plot
scatter = ax.scatter(top_spread_df['spatial_spread'], 
                    top_spread_df['dominant_node_pct'],
                    s=top_spread_df['total_samples']*3,  # Size by sample count
                    alpha=0.7,
                    c=top_spread_df['shannon_diversity'],
                    cmap='viridis',
                    edgecolors='black',
                    linewidth=1)

# Add site labels
for _, row in top_spread_df.iterrows():
    ax.annotate(row['site'], 
               (row['spatial_spread'], row['dominant_node_pct']),
               xytext=(5, 5),
               textcoords='offset points',
               fontsize=9,
               fontweight='bold' if row['site'] in top_spread_sites[:5] else 'normal',
               alpha=0.8)

ax.set_xlabel('Spatial Spread Index', fontsize=12)
ax.set_ylabel('Dominant Node Percentage (%)', fontsize=12)
ax.set_title(f'Top {N_TOP_SITES} Spread Sites: Spread vs Concentration\n(Size = Sample Count, Color = Shannon Diversity)', 
             fontweight='bold', fontsize=14)

# Add colorbar for diversity
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Shannon Diversity', fontsize=11)

# Add grid and threshold lines
ax.grid(True, alpha=0.3)

# Add median lines
median_spread = top_spread_df['spatial_spread'].median()
median_conc = top_spread_df['dominant_node_pct'].median()

ax.axvline(x=median_spread, color='r', 
          linestyle='--', alpha=0.5, linewidth=1, label=f'Median Spread ({median_spread:.2f})')
ax.axhline(y=median_conc, color='g', 
          linestyle='--', alpha=0.5, linewidth=1, label=f'Median Concentration ({median_conc:.1f}%)')

# Add quadrant labels
bbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax.text(0.05, 0.95, 'High Spread,\nLow Concentration',
       transform=ax.transAxes, fontsize=10, 
       bbox=bbox_props,
       verticalalignment='top')
ax.text(0.65, 0.95, 'High Spread,\nHigh Concentration',
       transform=ax.transAxes, fontsize=10,
       bbox=bbox_props,
       verticalalignment='top')

ax.legend(loc='lower right')
plt.tight_layout()

# Save Plot 2
for path in analysis_dirs:
    try:
        file_path = os.path.join(path, f"spread_vs_concentration_scatter.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"✓ Spread vs concentration scatter plot saved: {file_path}")
    except Exception as e:
        print(f"✗ Error saving to {path}: {e}")

plt.show(block=False)
plt.pause(1)
plt.close()

# ------------------------------------------------------------
# PLOT 3: Comparison of top spread vs top clustered sites
# ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# LEFT: Top spread sites
top_spread_sorted = top_spread_df.sort_values('spatial_spread', ascending=True)
bars1 = ax1.barh(range(len(top_spread_sorted)), top_spread_sorted['spatial_spread'])
colors1 = plt.cm.Blues(np.linspace(0.6, 0.9, len(top_spread_sorted)))
for i, bar in enumerate(bars1):
    bar.set_color(colors1[i])
    ax1.text(top_spread_sorted.iloc[i]['spatial_spread'] + 0.02, i,
            f"{top_spread_sorted.iloc[i]['spatial_spread']:.2f}",
            va='center', fontsize=9)

ax1.set_yticks(range(len(top_spread_sorted)))
ax1.set_yticklabels(top_spread_sorted['site'], fontsize=9)
ax1.set_xlabel('Spatial Spread Index', fontsize=11)
ax1.set_title(f'Top {N_TOP_SITES} Spatially Spread Sites', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3, axis='x')

# RIGHT: Top clustered sites
top_clustered_df = plot_df[plot_df['site'].isin(top_clustered_sites)].copy()
top_clustered_sorted = top_clustered_df.sort_values('spatial_spread', ascending=True)
bars2 = ax2.barh(range(len(top_clustered_sorted)), top_clustered_sorted['spatial_spread'])
colors2 = plt.cm.Reds(np.linspace(0.6, 0.9, len(top_clustered_sorted)))
for i, bar in enumerate(bars2):
    bar.set_color(colors2[i])
    ax2.text(top_clustered_sorted.iloc[i]['spatial_spread'] + 0.005, i,
            f"{top_clustered_sorted.iloc[i]['spatial_spread']:.2f}",
            va='center', fontsize=9)

ax2.set_yticks(range(len(top_clustered_sorted)))
ax2.set_yticklabels(top_clustered_sorted['site'], fontsize=9)
ax2.set_xlabel('Spatial Spread Index', fontsize=11)
ax2.set_title(f'Top {N_TOP_SITES} Clustered Sites', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')

plt.suptitle(f'Comparison: Spatially Spread vs Clustered Sites (Top {N_TOP_SITES})', 
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()

# Save Plot 3
for path in analysis_dirs:
    try:
        file_path = os.path.join(path, f"spread_vs_clustered_comparison.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"✓ Spread vs clustered comparison saved: {file_path}")
    except Exception as e:
        print(f"✗ Error saving to {path}: {e}")

plt.show(block=False)
plt.pause(1)
plt.close()

# ------------------------------------------------------------
# Save data summary
# ------------------------------------------------------------
for path in analysis_dirs:
    try:
        # Save top spread sites data
        top_spread_df.to_csv(os.path.join(path, f"top_{N_TOP_SITES}_spread_sites.csv"), index=False)
        
        # Save top clustered sites data
        top_clustered_df.to_csv(os.path.join(path, f"top_{N_TOP_SITES}_clustered_sites.csv"), index=False)
        
        print(f"✓ Data files saved to: {path}")
    except Exception as e:
        print(f"✗ Error saving data files to {path}: {e}")

# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------
print("\n" + "="*60)
print(f"TOP {N_TOP_SITES} SPATIAL SPREAD ANALYSIS")
print("="*60)

print(f"\nTop {N_TOP_SITES} Most Spatially Spread Sites:")
print("-" * 50)
for i, (_, row) in enumerate(top_spread_df.sort_values('spatial_spread', ascending=False).iterrows()):
    print(f"{i+1:2d}. {row['site']:20s}: Spread={row['spatial_spread']:.2f} | "
          f"Concentration={row['dominant_node_pct']:.1f}% | "
          f"Samples={row['total_samples']} | "
          f"Diversity={row['shannon_diversity']:.2f}")

print(f"\nTop {N_TOP_SITES} Most Clustered Sites:")
print("-" * 50)
for i, (_, row) in enumerate(top_clustered_df.sort_values('spatial_spread', ascending=True).iterrows()):
    print(f"{i+1:2d}. {row['site']:20s}: Spread={row['spatial_spread']:.2f} | "
          f"Concentration={row['dominant_node_pct']:.1f}% | "
          f"Samples={row['total_samples']} | "
          f"Diversity={row['shannon_diversity']:.2f}")

# Identify interesting patterns
print(f"\nINTERESTING PATTERNS:")
print("-" * 50)

# Sites with high spread AND high concentration
high_spread_high_conc = top_spread_df[top_spread_df['dominant_node_pct'] > 70]
if not high_spread_high_conc.empty:
    print(f"\n• Sites with BOTH high spread AND high concentration (>70%):")
    for _, row in high_spread_high_conc.iterrows():
        print(f"  {row['site']}: Spread={row['spatial_spread']:.2f}, "
              f"Concentration={row['dominant_node_pct']:.1f}%")

# Sites with high spread BUT low concentration
high_spread_low_conc = top_spread_df[top_spread_df['dominant_node_pct'] < 30]
if not high_spread_low_conc.empty:
    print(f"\n• Sites with high spread BUT low concentration (<30%):")
    for _, row in high_spread_low_conc.iterrows():
        print(f"  {row['site']}: Spread={row['spatial_spread']:.2f}, "
              f"Concentration={row['dominant_node_pct']:.1f}%")

#(6) Shannon diversity vs unique nodes
fig, ax = plt.subplots(figsize=(7, 6))

colors = ['red' if x > 70 else 'orange' if x > 50 else 'green'
          for x in plot_df['dominant_node_pct']]
ax.scatter(
    plot_df['shannon_diversity'],
    plot_df['unique_nodes'],
    c=colors,
    s=100,
    alpha=0.7
)
ax.set_xlabel('Node Diversity (Shannon)')
ax.set_ylabel('Unique Nodes')
ax.set_title('Node Diversity vs Node Richness')
for _, row in plot_df.iterrows():
    if row['site'] in notable_sites:
        ax.annotate(
            row['site'],
            (row['shannon_diversity'], row['unique_nodes']),
            xytext=(6, 6),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold'
        )
plt.tight_layout()
for path in analysis_dirs:
    file_path = os.path.join(path, "node_shannonvsunique.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Node diversity per site saved: {file_path}")

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

# ~ print_node_diversity_stats(site_stats)

# create site-node distribution matrix
def create_site_node_matrix(som, site_node_mapping):
    """Create a matrix showing site distribution across nodes"""
    global umatrix
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

# ~ create_site_node_matrix(som, site_node_mapping)

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

# ~ site_diversity_df = analyze_node_diversity_per_site(bmu_sites)
# ~ plot_site_node_diversity(site_diversity_df)

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
def analyze_alignment(node_cluster, bmu_locations, cluster, som_shape):
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
            # disagreement = fraction of samples whose sample_label != node_cluster[i,j]
            node_cluster_label = int(node_cluster[i,j])
            disagree = sum(1 for lab in labels_here if lab != node_cluster_label)/total

            node_stats.append({'node':(i,j),'cluster':total,'purity':purity,'entropy':ent,'majority_label':majority_label,'disagreement_rate':disagree,'counts':counts})
            all_node_majority[i,j] = majority_label
            all_node_purity[i,j] = purity
            all_node_entropy[i,j] = ent
            all_node_disagreement_rate[i,j] = disagree

    # global agreement: compare sample-level labels against node-level label of its BMU
    sample_node_labels = np.array([ node_cluster[x,y] for (x,y) in bmu_locations ])
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
result = analyze_alignment(node_cluster=node_cluster, bmu_locations=bmu_locations, cluster=df2['cluster'], som_shape=som_shape)
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
def plot_pies_on_nodes(node_stats, node_cluster, som_shape, n_clusters=n_clusters):
    """Plot small pies showing cluster composition per node with consistent colors"""
    # UNPACK the som_shape tuple
    if isinstance(som_shape, (int, float)):
        nx = ny = int(som_shape)  # Square grid
    else:
        nx, ny = som_shape  # Unpack tuple
    
    # consistent cluster coloring
    tab20 = plt.get_cmap('tab20')
    custom_indices = [0, 5, 10, 15, 19, 14, 3]
    cluster_colors = [tab20(i) for i in custom_indices[:n_clusters]]
    
    # Create a mapping from cluster ID to color
    # Handle case where cluster IDs don't start from 0 or aren't sequential
    cluster_color_map = {}
    
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
    
    # First pass: collect all unique cluster IDs to build color mapping
    all_cluster_ids = set()
    
    if isinstance(node_stats, dict):
        for (i, j), stats in node_stats.items():
            if isinstance(stats, dict) and 'cluster_counts' in stats:
                counts = stats['cluster_counts']
            elif isinstance(stats, dict):
                counts = stats
            else:
                continue
                
            if counts and isinstance(counts, dict):
                all_cluster_ids.update(counts.keys())
    elif isinstance(node_stats, pd.DataFrame):
        for _, row in node_stats.iterrows():
            counts = None
            for col in ['counts', 'cluster_counts', 'cluster_composition']:
                if col in row and isinstance(row[col], dict):
                    counts = row[col]
                    break
            
            if counts and isinstance(counts, dict):
                all_cluster_ids.update(counts.keys())
    
    # Sort cluster IDs and create color mapping
    sorted_cluster_ids = sorted(all_cluster_ids)
    n_unique_clusters = len(sorted_cluster_ids)
    
    # Extend colors if needed or use color cycle
    if n_unique_clusters > len(cluster_colors):
        # If more clusters than colors, extend with additional colors
        cmap = plt.cm.get_cmap('tab20', max(n_unique_clusters, 20))
        cluster_colors = [cmap(i) for i in range(n_unique_clusters)]
    
    # Create mapping from cluster ID to color
    for idx, cluster_id in enumerate(sorted_cluster_ids):
        cluster_color_map[cluster_id] = cluster_colors[idx % len(cluster_colors)]
    
    print(f"Found {n_unique_clusters} unique clusters: {sorted_cluster_ids}")
    print(f"Using {len(cluster_colors)} colors, n_clusters parameter = {n_clusters}")
    
    # Second pass: plot the pies
    if isinstance(node_stats, dict):
        # If node_stats is a dictionary with (x,y) keys
        for (i, j), stats in node_stats.items():
            # Check if stats has cluster_counts or if it's the counts directly
            if isinstance(stats, dict) and 'cluster_counts' in stats:
                counts = stats['cluster_counts']
            elif isinstance(stats, dict):
                counts = stats  # stats is the counts dictionary itself
            else:
                continue
                
            if not counts or not isinstance(counts, dict):
                continue
                
            # prepare data for pie chart
            labels = list(counts.keys())
            sizes = list(counts.values())
            
            if not labels:  # Skip if no data
                continue
            
            # map cluster numbers to colors using the mapping
            try:
                pie_colors = [cluster_color_map[cluster_id] for cluster_id in labels]
            except KeyError as e:
                print(f"Warning: Cluster ID {e} not found in color mapping. Using default colors.")
                # Use color cycle for missing clusters
                cmap = plt.cm.get_cmap('tab20')
                pie_colors = [cmap(i % 20) for i in range(len(labels))]
            
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
            
            if not labels:  # Skip if no data
                continue
            
            # Map cluster numbers to colors using the mapping
            try:
                pie_colors = [cluster_color_map[cluster_id] for cluster_id in labels]
            except KeyError as e:
                print(f"Warning: Cluster ID {e} not found in color mapping. Using default colors.")
                cmap = plt.cm.get_cmap('tab20')
                pie_colors = [cmap(i % 20) for i in range(len(labels))]
            
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

    # legend with the same clustering colors
    # Create legend items for all unique clusters found
    legend_elements = [
        mpatches.Patch(color=cluster_color_map[cluster_id], label=f'Cluster {cluster_id}')
        for cluster_id in sorted_cluster_ids
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
    
plot_pies_on_nodes(result['node_stats'], node_cluster, som_shape)

'''
# overlay misaligned nodes on umatrix
plt.imshow(umatrix, cmap='bone_r')
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
plt.close()'''

# summary
summary = result['node_stats'][['purity', 'entropy', 'disagreement_rate']].describe()
print(summary)

"""Node-level purity ranged from 0.42–1.0 (mean = 0.78), while disagreement rates averaged 0.18 ± 0.12, suggesting moderate structural consistency between BMU- and node-level cluster representations."""

# ----------------------------------------------------------------------
# Within-node homogeneity - how similar the sites mapped to the same SOM node
# ----------------------------------------------------------------------
def compute_within_node_distances(df, som, X_hel, metrics=["euclidean", "cosine", "cityblock"]):
    """Compute within-node distances"""
    node_samples = defaultdict(list)
    
    for idx, x in enumerate(X_hel):
        node = som.winner(x)
        node_samples[node].append(idx)
    
    results = []
    for node, idxs in node_samples.items():
        if len(idxs) < 2:
            continue
        data = X_hel[idxs]
        
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

# ~ within_df = compute_within_node_distances(df2, som, X_hel)

'''
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
plt.close()'''

'''
Each node has an average intra-node distance per metric.
Lower mean distance = more homogeneous sites within that node.
Compare across metrics:
    manhattan: absolute spread.
    manhattan: angular spread (direction similarity).
    Mahalanobis: adjusted for correlations in variables.
'''
# ----------------------------------------------------------------------
# Between-node distances (SOM weights centroids) - how far apart the SOM node centroids (weights) are in feature space
# ----------------------------------------------------------------------
def compute_between_node_distances(weights, metrics=["euclidean", "cosine", "cityblock"]):
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
# Compute BMU-wise averages
# ----------------------------------------------------------------------
'''def compute_bmu_averages(df, X_hel, som, species_vars, chemical_vars):
    n_nodes_x, n_nodes_y = som._weights.shape[:2]
    bmu_species = np.zeros((n_nodes_x, n_nodes_y, len(species_vars)))
    bmu_chemicals = np.zeros((n_nodes_x, n_nodes_y, len(chemical_vars)))
    bmu_counts = np.zeros((n_nodes_x, n_nodes_y))

    for idx, row in df.iterrows():
        x, y = som.winner(X_hel[idx])
        bmu_species[x, y] += row[species_vars].fillna(0).values
        bmu_chemicals[x, y] += row[chemical_vars].fillna(0).values
        bmu_counts[x, y] += 1

    # Average over counts
    bmu_species_avg = np.divide(bmu_species, bmu_counts[..., np.newaxis], out=np.zeros_like(bmu_species), where=bmu_counts[..., np.newaxis]!=0)
    bmu_chemicals_avg = np.divide(bmu_chemicals, bmu_counts[..., np.newaxis], out=np.zeros_like(bmu_chemicals), where=bmu_counts[..., np.newaxis]!=0)

    return bmu_species_avg, bmu_chemicals_avg

bmu_species_avg, bmu_chemicals_avg = compute_bmu_averages(df1, X_hel, som, major_cats, env_vars)'''


def compute_bmu_averages(df, X_hel, som, sp_vars, env_vars):
    n_nodes_x, n_nodes_y = som.get_weights().shape[:2]
    bmu_species = np.zeros((n_nodes_x, n_nodes_y, len(sp_vars)))
    bmu_envs = np.zeros((n_nodes_x, n_nodes_y, len(env_vars)))
    bmu_counts = np.zeros((n_nodes_x, n_nodes_y))

    for idx in range(len(df)):
        x, y = som.winner(X_hel[idx])
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

# ~ bmu_sp_avg, bmu_env_avg = compute_bmu_averages(df2, X_hel, som, sp_vars, env_vars)

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

