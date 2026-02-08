## =====================================================================
## Objective: Analyze trained SOM using benthic cover (not per site) to uncover gradients in composition space and link them to environmental gradient
## Output: U-Matrix, component planes/overlays, hitmaps
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

# Hitmap analysis: count how many samples fall on each BMU
neuron_counts = np.zeros((xdim, ydim), dtype=int)
neuron_counts = np.array(neuron_counts)

for (i, j) in bmu_locations:
    neuron_counts[i, j] += 1
    
plt.figure(figsize=(10, 8))
im = plt.imshow(neuron_counts, origin='lower', cmap='viridis')
plt.colorbar(im, label='Number of samples (hits)')
plt.title('SOM Hit Map (Sample Density per Node)')
plt.xlabel('SOM Y')
plt.ylabel('SOM X')

for path in analysis_dirs:
    plt.savefig(os.path.join(path, "som_hit_map.png"), dpi=300, bbox_inches='tight')

plt.show(block=False)
plt.pause(1)
plt.close()

# ----------------------------------------------------------------------
# Overlay sample counts on U-Matrix
# ----------------------------------------------------------------------
def overlay_hits_on_umatrix(som, neuron_counts, result_dirs=None):

    umatrix, xdim, ydim
   
    plt.figure(figsize=(12, 10))
    
    plt.imshow(umatrix, cmap='bone_r', origin='lower', alpha=0.7)
    plt.colorbar(label="Distance")
    plt.title('SOM U-Matrix with Samples per Node')

    # get dimensions from node_predicted_cluster_map
    xdim_map, ydim_map = neuron_counts.shape
    print(f"Node cluster map shape: {neuron_counts.shape}")
    print(f"Should match U-Matrix: {umatrix.shape}")
    
    # overlay cluster numbers
    for i in range(xdim):
        for j in range(ydim):
            cluster_num = neuron_counts[i, j]
            plt.text(j, i, str(cluster_num), 
                     ha='center', va='center',
                     color='black', fontsize=12
                     )
    
    plt.tight_layout()
    
    if result_dirs:
        for path in analysis_dirs:
            try:
                os.makedirs(path, exist_ok=True)
                file_path = os.path.join(path, "umatrix_hitmap_overlay.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                print(f"✓ U-Matrix with hits overlay saved: {file_path}")
            except Exception as e:
                print(f"✗ Failed to save to {path}: {e}")
    
    plt.show(block=False)
    plt.pause(1)
    plt.close()

overlay_hits_on_umatrix(som, neuron_counts, result_dirs)

def create_normalized_hitmap(neuron_counts, result_dirs=None):

    total_samples = neuron_counts.sum()
    normalized_hits = neuron_counts.astype(float) / total_samples * 100
    
    plt.figure(figsize=(12, 10))
    
    im = plt.imshow(normalized_hits, origin='lower', cmap='plasma')
    cbar = plt.colorbar(im, label='Percentage of Total Samples (%)', shrink=0.8)
    cbar.formatter.set_powerlimits((0, 0))
    
    plt.title('Normalized SOM Hit Map (% of Total Samples)', fontsize=14, pad=15)
    plt.xlabel('SOM Y Dimension', fontsize=12)
    plt.ylabel('SOM X Dimension', fontsize=12)
    
    # add percentage labels
    for i in range(xdim):
        for j in range(ydim):
            if normalized_hits[i, j] > 0.1:  # Show if > 0.1%
                plt.text(j, i, f"{normalized_hits[i, j]:.1f}%", 
                        ha='center', va='center',
                        color='white' if normalized_hits[i, j] > np.median(normalized_hits[normalized_hits > 0]) else 'black',
                        fontsize=8, fontweight='bold')
    
    
    plt.tight_layout()
    
    if result_dirs:
        for path in analysis_dirs:
            try:
                os.makedirs(path, exist_ok=True)
                file_path = os.path.join(path, "umatrix_normalized_hitmap_overlay.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                print(f"✓ U-Matrix with hits overlay saved: {file_path}")
            except Exception as e:
                print(f"✗ Failed to save to {path}: {e}")
    
    plt.show(block=False)
    plt.pause(1)
    plt.close()

create_normalized_hitmap(neuron_counts, result_dirs)

# ----------------------------------------------------------------------
# Neighborhood Preservation Index
# ----------------------------------------------------------------------
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

som_coords = np.array([som.winner(x) for x in X_hel])
npi_score = neighborhood_preservation_index(X_hel, som_coords, k=5)
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
    """SOM visualizations: U-Matrix and component planes (individual)"""
    global umatrix
    plt.figure(figsize=(12, 10))
    plt.imshow(umatrix, cmap='bone_r', origin='lower')
    plt.colorbar(label="Distance")
    plt.title('SOM U-Matrix (manhattan Distance)')

    # grid coordinates for reference
    plt.xticks(np.arange(umatrix.shape[1]))
    plt.yticks(np.arange(umatrix.shape[0]))
    # ~ plt.grid(True, alpha=0.3, linestyle='--')
    
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

        for path in analysis_dirs:
            try:
                os.makedirs(path, exist_ok=True)
                file_path = os.path.join(path, f"component_plane_{var}.png")
                plt.savefig(file_path, dpi=150)
                print(f"Component plane for {var} saved to: {file_path}")
            except Exception as e:
                print(f"Failed to save component plane for {var} to {path}: {e}")
        plt.close()
'''
visualize_som(som, variables_of_interest, analysis_dirs)

# ----------------------------------------------------------------------
# Biodiversity analysis
# ----------------------------------------------------------------------
# calculate species richness
df2['Species_Richness'] = df2[sp_vars].gt(0).sum(axis=1)

# richness vs. SOM clusters
plt.figure(figsize=(12, 6))
sns.boxplot(data=df2, x='SOM_Cluster', y='Species_Richness', palette=discrete_color)
plt.title('Species Richness Across manhattan-based SOM Clusters')
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
exit()
# ----------------------------------------------------------------------
# Major component along SOM
# ----------------------------------------------------------------------

som_x, som_y = 29, 27

major_cats = [ 'Ascidian', 'CCA', 'Corallimorpharia', 'Gorgonian', 'Hard coral', 'Macroalgae', 'Other mobile invertebrates', 'Other sessile invertebrates', 'Seagrass', 'Soft coral', 'Sponge',  'Turf', 'Zoanthid']

# prelocated species map
species_map = [np.zeros((som_x, som_y)) for _ in range(len(major_cats))]
neuron_counts = np.zeros((som_x, som_y)) # create single array to track sample count per neuron

# (2) aggregate species data to BMU locations
for idx, (i, j) in enumerate(bmu_locations):
    neuron_counts[i, j] += 1
    row = df2[major_cats].iloc[idx].values
    for k in range(len(major_cats)):
        species_map[k][i, j] += row[k]
        
# (2) compute average cover per SOM unit per species
for k in range(len(major_cats)):
    species_map[k] = np.divide(
        species_map[k], 
        neuron_counts, 
        out=np.zeros_like(species_map[k]),
        where=neuron_counts != 0
    )

# (2) plot overlay for species
fig, axes = plt.subplots(4, 5, figsize=(18, 10))
for i, ax in enumerate(axes.flat):
    if i < len(major_cats):
        # mask zero values
        masked_data = np.ma.masked_where(species_map[i].T == 0,
                                         species_map[i].T)
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
# Major component along SOM
# ----------------------------------------------------------------------

som_x, som_y = 29, 27

major_cats = [ 'Ascidian', 'CCA', 'Corallimorpharia', 'Gorgonian', 'Hard coral', 'Macroalgae', 'Other mobile invertebrates', 'Other sessile invertebrates', 'Seagrass', 'Soft coral', 'Sponge',  'Turf', 'Zoanthid']

# prelocated species map
species_map = [np.zeros((som_x, som_y)) for _ in range(len(major_cats))]
neuron_counts = np.zeros((som_x, som_y)) # create single array to track sample count per neuron

# (2) aggregate species data to BMU locations
for idx, (i, j) in enumerate(bmu_locations):
    neuron_counts[i, j] += 1
    row = df2[major_cats].iloc[idx].values
    for k in range(len(major_cats)):
        species_map[k][i, j] += row[k]
        
# (2) compute average cover per SOM unit per species
for k in range(len(major_cats)):
    species_map[k] = np.divide(
        species_map[k], 
        neuron_counts, 
        out=np.zeros_like(species_map[k]),
        where=neuron_counts != 0
    )

# (2) plot overlay for species
fig, axes = plt.subplots(4, 5, figsize=(18, 10))
for i, ax in enumerate(axes.flat):
    if i < len(major_cats):
        # mask zero values
        masked_data = np.ma.masked_where(species_map[i].T == 0,
                                         species_map[i].T)
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
# Overlay sites on U-Matrix
# ----------------------------------------------------------------------
def overlay_site_names_on_som(som, df, label_col='site'):
    global umatrix
    plt.figure(figsize=(12, 10))
    plt.imshow(umatrix, cmap='bone_r', origin='lower')  # SAME orientation
    plt.colorbar(label="Distance")
    plt.title('manhattan-based U-Matrix with Site Labels')

    # BMU assignment
    bmu_labels = defaultdict(list)
    for idx, row in df.iterrows():
        x, y = som.winner(X_hel[idx])
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

def overlay_site_sample_on_som(som, df, X_hel, label_col='site', mode='count'):
    """
    mode = 'count'  -> show only n per node
    mode = 'site'   -> show only site names
    mode = 'both'   -> show site + n
    """
    global umatrix
    plt.figure(figsize=(12, 10))
    plt.imshow(umatrix, cmap='bone_r', origin='lower')
    plt.colorbar(label="Distance")
    plt.title("U-Matrix Overlay")

    # Group samples by BMU
    bmu_labels = defaultdict(list)
    for idx, row in df.iterrows():
        x, y = som.winner(X_hel[idx])
        bmu_labels[(x, y)].append(str(row[label_col]))

    for (x, y), labels in bmu_labels.items():
        count = len(labels)
        unique_labels = sorted(set(labels))
        label_text = ", ".join(unique_labels)
        if len(label_text) > 15:
            label_text = label_text[:12] + "..."

        if mode == 'count':
            display_text = f"{count}"
        elif mode == 'site':
            display_text = label_text
        else:  # both
            display_text = f"{label_text}\n(n={count})"

        plt.text(
            y, x, display_text,
            color='red', fontsize=8, fontweight='bold',
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
        )

    plt.tight_layout()
    for path in analysis_dirs:
        file_path = os.path.join(path, f"u_matrix_overlay_{label_col}_{mode}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Overlay plot saved: {file_path}")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

overlay_site_sample_on_som(som, df2, X_hel, mode='count')

# ----------------------------------------------------------------------
# Classification of samples based on the label assigned to the associated winning neuron
# ----------------------------------------------------------------------
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
        X_hel = training_data['X_hel']
        # ~ variables_of_interest = training_data['variables_of_interest']
        bmu_locations = training_data['bmu_locations']
        
    except Exception as e:
        print(f"Error loading training output: {e}")
        sys.exit(1)

    # prepare data
    data = X_hel
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
# U-Matrix + selected species overlays : Species hitmap
# ----------------------------------------------------------------------
def plot_species_overlays(som, df, species_list, X_hel, n_cols=4):
    n_categories = len(species_list)
    n_panels = n_categories + 1  # +1 for U-Matrix
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Panel A: U-Matrix
    axes[0].imshow(som.distance_map(), cmap='bone_r', origin='lower')
    axes[0].set_title('manhattan-based U-Matrix')

    for i, species in enumerate(species_list):
        species_density = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1]))

        for idx, row in df.iterrows():
            x, y = som.winner(X_hel[idx])
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

# ~ plot_species_overlays(som, df2, ['pescca', 'othcca', 'monfol', 'monenc', 'tursed', 'turfil', 'pocbus'], X_hel) # check kmeans+pca for ost pronounced features
# ~ plot_species_overlays(som, df2, sp_vars, X_hel)

