import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import os
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from dotenv import load_dotenv

load_dotenv()

class _GTZANLoader:
    def __init__(self) -> None:
        """
        Loads and preprocesses the GTZAN dataset from a CSV file.
        Expects the CSV path and manifold directory to be set in environment variables.
        """

        self.GTZAN_CSV = os.getenv("GTZAN_CSV")
        self.MANIFOLD_DIR = os.getenv("MANIFOLD_DIR")

        if not self.MANIFOLD_DIR:
            raise ValueError("MANIFOLD_DIR path not found in environment variables.")
        self.manifold_dir = os.path.join(os.getcwd(), self.MANIFOLD_DIR)
        if not self.GTZAN_CSV:
            raise ValueError("GTZAN_CSV path not found in environment variables.")
        self.gtzan_csv = os.path.join(os.getcwd(), self.GTZAN_CSV)
        if not os.path.isfile(self.gtzan_csv):
            raise FileNotFoundError(f"GTZAN CSV file not found at path: {self.gtzan_csv}")
        print("Loading GTZAN CSV...")

        self.data = pd.read_csv(self.gtzan_csv)
        if self.data.empty:
            raise ValueError("GTZAN CSV file is empty or could not be read.")
        
        self.non_feature_cols = ["filename", "label", "length"]
        self.features = self.data.drop(columns=self.non_feature_cols).values
        self.labels = self.data["label"]

        self.label_encoder = LabelEncoder()
        print("Encoding labels...")
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        print("Scaling features...")
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)

def cluster_purity(embedding, labels, n_neighbors=10):
    """Calculate average local cluster purity based on k-nearest neighbors.
     For each point, find its k nearest neighbors and compute the fraction that 
     share the same label.
    Args:
        embedding (np.ndarray): The data points in the embedded space.
        labels (np.ndarray): The true labels for each data point.
        n_neighbors (int): Number of neighbors to consider for purity calculation.
    Returns:
        float: Average local cluster purity.
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors+1).fit(embedding)
    distances, indices = nn.kneighbors(embedding)
    correct = 0
    for i in range(len(embedding)):
        neigh_labels = labels[indices[i][1:]]  
        main_label = labels[i]
        correct += np.sum(neigh_labels == main_label) / n_neighbors
    return correct / len(embedding)

def genre_neighbor_matrix(embedding, labels, label_encoder, n_neighbors=10):
    """Create a genre neighbor matrix showing how often genres appear in 
    each other's neighborhoods.
    
    Args:
        embedding (np.ndarray): The data points in the embedded space.
        labels (np.ndarray): The true labels for each data point.
        label_encoder (LabelEncoder): The label encoder used to encode the labels.
        n_neighbors (int): Number of neighbors to consider.
    Returns:
        pd.DataFrame: A DataFrame representing the genre neighbor matrix.
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors+1).fit(embedding)
    _, indices = nn.kneighbors(embedding)
    matrix = np.zeros((len(label_encoder.classes_), len(label_encoder.classes_)), dtype=int)
    for i in range(len(embedding)):
        genre = labels[i]
        for j in indices[i][1:]:  
            matrix[genre, labels[j]] += 1
    df = pd.DataFrame(matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    return df

def run_umap_sweep(loader, n_neighbors_list, min_dist_list, n_components=50):
    """Run a sweep over UMAP hyperparameters and evaluate clustering metrics.
    Args:
        loader (_GTZANLoader): The data loader with preprocessed features and labels.
        n_neighbors_list (list): List of n_neighbors values to try.
        min_dist_list (list): List of min_dist values to try.
        n_components (int): Number of UMAP components (dimensions).
    Returns:
        pd.DataFrame: A DataFrame summarizing the results of the sweep.
    """
    labels = loader.encoded_labels
    label_encoder = loader.label_encoder

    sweep_results = []
    
    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            print(f"\n==== UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components} ====")
            umap_model = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist
            )
            embedding = np.array(umap_model.fit_transform(loader.scaled_features))

            print("Any NaNs?", np.isnan(embedding).any())
            print("Any infs?", np.isinf(embedding).any())

            sil = silhouette_score(embedding, labels)
            ch = calinski_harabasz_score(embedding, labels)
            db = davies_bouldin_score(embedding, labels)
            purity = cluster_purity(embedding, labels, n_neighbors=10)

            print(f"Silhouette Score: {sil:.3f}")
            print(f"Calinski-Harabasz Score: {ch:.3f}")
            print(f"Davies-Bouldin Score: {db:.3f}")
            print(f"Average local cluster purity (k=10): {purity:.3f}")

            genre_matrix = genre_neighbor_matrix(embedding, labels, label_encoder, n_neighbors=10)
            
            print("Genre neighbor matrix (k=10):")
            print(genre_matrix)
            
            plt.figure(figsize=(10,8))
            sns.heatmap(genre_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Genre neighbor matrix (k=10): n_neighbors={n_neighbors}, min_dist={min_dist}, n_comp={n_components}")
            plt.ylabel("Anchor Genre")
            plt.xlabel("Neighbor Genre")
            plt.tight_layout()
            plt.show()

            kmeans = KMeans(n_clusters=len(label_encoder.classes_), random_state=42)
            cluster_labels = kmeans.fit_predict(embedding)
            ari = adjusted_rand_score(labels, cluster_labels)
            print(f"KMeans cluster ARI vs. true genres: {ari:.3f}")

            sweep_results.append({
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "purity": purity,
                "kmeans_ARI": ari
            })
    return pd.DataFrame(sweep_results)

if __name__ == "__main__":
    print("Preparing GTZAN data...")
    loader = _GTZANLoader()

    n_neighbors_list = [5, 10, 20, 30, 40, 50]
    min_dist_list = [0.01, 0.05, 0.1, 0.2, 0.5]

    n_components = 50

    results_df = run_umap_sweep(loader, n_neighbors_list, min_dist_list, 
                                n_components=n_components)
    print("\n\n==== Sweep Summary Table ====\n")
    print(results_df)
        