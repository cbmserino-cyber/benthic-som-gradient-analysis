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
import os
import numpy as np
import pandas as pd
from joblib import dump, load
import pickle
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from seaborn import scatterplot
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib as mpl
from numpy import array, cov, mean, setdiff1d, sqrt, linspace
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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

som_weights = som.get_weights()    
x = som_weights.reshape(-1, som_weights.shape[2])

# ----------------------------------------------------------------------
# Build SOM node dataset weighted by hit counts
# ----------------------------------------------------------------------
som_weights = som.get_weights()
X_nodes = som_weights.reshape(-1, som_weights.shape[2])

# compute hit counts per node
map_size = som_weights.shape[:2]
hit_counts = np.zeros(map_size, dtype=int)

for bmu in bmu_locations:
    hit_counts[tuple(bmu)] += 1

node_hits = hit_counts.reshape(-1)

# keep only nodes that are actually occupied
mask = node_hits > 0
Xw = X_nodes[mask]
Hw = node_hits[mask]   # sample weights

def elbow_method(X, max_k=15):
    """Elbow method using Within-Cluster Sum of Squares (WCSS)"""
    wcss = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(Xw, sample_weight=Hw)
        wcss.append(kmeans.inertia_)
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, 'bo-', alpha=0.7)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(k_range)
    plt.grid(True, alpha=0.3)
    plt.savefig('elbow_method.png')
    plt.show(block=False)
    plt.pause(1)
    plt.close(fig)

    optimal_k = k_range[np.argmax(wcss)]
    print(f"Optimal k from wcss: {optimal_k}")
    return wcss

def silhouette_analysis(X, max_k=15):
    """Silhouette score analysis"""
    silhouette_scores = []
    k_range = range(2, max_k + 1)  # k=1 not valid for silhouette
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(Xw, sample_weight=Hw)
        X_exp = np.repeat(Xw, Hw, axis=0)
        lab_exp = np.repeat(labels, Hw)
        score = silhouette_score(X_exp, lab_exp)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'ro-', alpha=0.7)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.xticks(k_range)
    plt.grid(True, alpha=0.3)
    plt.savefig('silhouette_analysis.png')
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal k from silhouette: {optimal_k}")
    return silhouette_scores, optimal_k

def gap_statistic(X, max_k=15, n_refs=10):
    """Gap statistic method"""
    def calculate_wcss(data, k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(Xw, sample_weight=Hw)
        return kmeans.inertia_
    
    gaps = []
    wcss = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        # Actual data WCSS
        actual_wcss = calculate_wcss(X, k)
        wcss.append(actual_wcss)
        
        # Reference WCSS (uniform random data)
        ref_wcss = []
        for i in range(n_refs):
            # Generate reference dataset
            random_data = np.random.uniform(
                low=X.min(axis=0), 
                high=X.max(axis=0), 
                size=X.shape
            )
            ref_wcss.append(calculate_wcss(random_data, k))
        
        # Calculate gap
        gap = np.log(np.mean(ref_wcss)) - np.log(actual_wcss)
        gaps.append(gap)
    
    # Find optimal k (first k where gap(k) ≥ gap(k+1) - std(k+1))
    optimal_k = 1
    for k in range(len(gaps) - 1):
        if gaps[k] >= gaps[k + 1] - np.std([gaps[k + 1]]):
            optimal_k = k + 1
            break
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(k_range, wcss, 'bo-', alpha=0.7)
    plt.title('WCSS vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(k_range, gaps, 'go-', alpha=0.7)
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
    plt.title('Gap Statistic')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Gap')
    plt.grid(True, alpha=0.3)
    plt.savefig('gap_statistic.png')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    
    print(f"Optimal k from gap statistic: {optimal_k}")
    return gaps, optimal_k

def calinski_harabasz_analysis(X, max_k=15):
    """Calinski-Harabasz index (Variance Ratio Criterion)"""
    ch_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(Xw, sample_weight=Hw)
        X_exp = np.repeat(Xw, Hw, axis=0)
        lab_exp = np.repeat(labels, Hw)
        score = calinski_harabasz_score(X_exp, lab_exp)
        ch_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, ch_scores, 'mo-', alpha=0.7)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Analysis for Optimal k')
    plt.xticks(k_range)
    plt.grid(True, alpha=0.3)
    plt.savefig('calinski_harabasz_analysis.png')
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    
    optimal_k = k_range[np.argmax(ch_scores)]
    print(f"Optimal k from Calinski-Harabasz: {optimal_k}")
    return ch_scores, optimal_k

def comprehensive_cluster_analysis(X, max_k=15):
    """Run all methods and find consensus optimal k"""
    methods = {}
    
    print("🔍 Running comprehensive cluster analysis...")
    
    # 1. Elbow Method
    print("\n1. Running Elbow Method...")
    wcss = elbow_method(X, max_k)
    
    # 2. Silhouette Analysis
    print("\n2. Running Silhouette Analysis...")
    silhouette_scores, silhouette_k = silhouette_analysis(X, max_k)
    methods['Silhouette'] = silhouette_k
    
    # 3. Gap Statistic
    print("\n3. Running Gap Statistic...")
    gaps, gap_k = gap_statistic(X, max_k)
    methods['Gap'] = gap_k
    
    # 4. Calinski-Harabasz
    print("\n4. Running Calinski-Harabasz Analysis...")
    ch_scores, ch_k = calinski_harabasz_analysis(X, max_k)
    methods['Calinski-Harabasz'] = ch_k
    
    # 5. Davies-Bouldin Index
    print("\n5. Running Davies-Bouldin Analysis...")
    db_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = davies_bouldin_score(X, labels)
        db_scores.append(score)
    
    db_k = k_range[np.argmin(db_scores)]  # Lower is better
    methods['Davies-Bouldin'] = db_k
    
    # Plot Davies-Bouldin
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, db_scores, 'co-', alpha=0.7)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Score (lower is better)')
    plt.title('Davies-Bouldin Analysis for Optimal k')
    plt.xticks(k_range)
    plt.grid(True, alpha=0.3)
    plt.savefig('comprehensive_cluster_analysis.png')
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    
    # Find consensus
    k_counts = {}
    for method, k in methods.items():
        k_counts[k] = k_counts.get(k, 0) + 1
    
    consensus_k = max(k_counts, key=k_counts.get)
    
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE CLUSTER ANALYSIS RESULTS")
    print("="*60)
    for method, k in methods.items():
        print(f"  {method:20}: k = {k}")
    print(f"\n🎯 CONSENSUS OPTIMAL k: {consensus_k}")
    
    # Show final clustering with optimal k
    final_kmeans = KMeans(n_clusters=consensus_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(X)
    final_silhouette = silhouette_score(X, final_labels)
    
    print(f"\n📈 Final clustering with k={consensus_k}:")
    print(f"   Silhouette score: {final_silhouette:.3f}")
    print(f"   Cluster sizes: {np.bincount(final_labels)}")
    
    return consensus_k, methods, final_labels

print("🚀 Starting optimal cluster analysis...")
print(f"Dataset shape: {x.shape}")
# ~ print(f"Features: {len(feature_labels)}")

optimal_k, all_methods, cluster_labels = comprehensive_cluster_analysis(Xw)
