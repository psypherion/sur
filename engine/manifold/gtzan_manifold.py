import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import joblib
import numpy as np
from dotenv import load_dotenv
import os
from tqdm import tqdm
from matplotlib import animation

load_dotenv()

class _GTZANLoader:
    """
    Load and preprocess GTZAN dataset from CSV.
    Expects environment variables:
    - GTZAN_CSV: Path to the GTZAN CSV file.
    - MANIFOLD_DIR: Directory to save manifold outputs.
    """
    def __init__(self) -> None:
        """Initialize and load GTZAN data.
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

class GTZANManifold:
    def __init__(self, n_components: int, model_name: str, 
                 n_neighbors=10, min_dist=0.1):
        """
        Initialize UMAP manifold for GTZAN dataset.
        Args:
            n_components (int): Number of dimensions for UMAP.
            model_name (str): Name for saving the model and embeddings.
            n_neighbors (int): UMAP n_neighbors parameter.
            min_dist (float): UMAP min_dist parameter.
        """
        self.loader = _GTZANLoader()
        self.manifold_dir = self.loader.manifold_dir
        self.n_components = n_components
        self.model_name = model_name
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.umap_model = None
        self.embedding = None

    def fit(self):
        """
        Fit UMAP model to the GTZAN scaled features.
        """
        print(f"Fitting UMAP ({self.n_components}D) manifold [n_neighbors={self.n_neighbors}, min_dist={self.min_dist}]: this may take a few moments...")
        with tqdm(total=1, desc=f"UMAP {self.n_components}D") as pbar:
            self.umap_model = umap.UMAP(
                n_components=self.n_components, 
                n_neighbors=self.n_neighbors, 
                min_dist=self.min_dist
            )

            self.embedding = self.umap_model.fit_transform(
                self.loader.scaled_features)
            pbar.update(1)

    def save(self):
        """
        Save the UMAP model and embeddings to files.
        """
        joblib.dump(self.umap_model, os.path.join(self.manifold_dir, 
                                                  f"{self.model_name}.pkl"))
        col_names = [f"UMAP_{i+1}" for i in range(self.n_components)]
        emb_df = pd.DataFrame(np.array(self.embedding), columns=col_names)
        emb_df['label_id'] = self.loader.encoded_labels
        emb_df['label'] = self.loader.labels.values
        emb_df.to_csv(os.path.join(self.manifold_dir, 
                                   f"{self.model_name}_embedding.csv"), 
                                   index=False)

def plot_umap_2d(embedding, labels, label_encoder, out_path=None):
    """
    Plot 2D UMAP embedding.
    Args:
        embedding (np.ndarray): 2D UMAP embedding.
        labels (np.ndarray): Encoded labels.
        label_encoder (LabelEncoder): Fitted label encoder.
        out_path (str, optional): Path to save the plot image.
    """
    plt.figure(figsize=(10,7))
    cmap = plt.get_cmap('tab10', len(label_encoder.classes_))
    for idx, genre in enumerate(label_encoder.classes_):
        mask = labels == idx
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1],
            color=cmap(idx), label=genre, s=40, alpha=0.7
        )
    plt.title('GTZAN UMAP Audio Manifold (2D)')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize=10)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.show()

def plot_umap_3d(embedding, labels, 
                 label_encoder, interactive=True, 
                 out_path=None, animate_path=None):
    """
    Plot and optionally animate 3D UMAP embedding.
    Args:
        embedding (np.ndarray): 3D UMAP embedding.
        labels (np.ndarray): Encoded labels.
        label_encoder (LabelEncoder): Fitted label encoder.
        interactive (bool): Whether to show the plot interactively.
        out_path (str, optional): Path to save the static plot image.
        animate_path (str, optional): Path to save the animated GIF.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('tab10', len(label_encoder.classes_))
    for idx, genre in enumerate(label_encoder.classes_):
        mask = labels == idx
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
            color=cmap(idx), label=genre, s=40, alpha=0.7
        )

    ax.set_title('GTZAN UMAP Audio Manifold (3D)')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_zlabel('UMAP-3')
    ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize=10)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)
    if interactive:
        plt.show()

    if animate_path:
        def rotate(angle):
            ax.view_init(elev=20, azim=angle)
            return fig.axes 
        rot_animation = animation.FuncAnimation(fig, rotate, 
                                                frames=np.arange(0, 360, 2), 
                                                interval=50)
        
        rot_animation.save(animate_path, dpi=80, writer='pillow')
        print(f"Saved animation to {animate_path}")

if __name__ == "__main__":
    out_dir = os.path.join(os.getcwd(),"engine", "manifold", "audio_manifold")
    print("Preparing GTZAN data...")
    loader = _GTZANLoader()

    vis2d = GTZANManifold(n_components=2, model_name="gtzan_umap_vis_2d",
                           n_neighbors=10, min_dist=0.1)
    vis2d.fit()
    vis2d.save()

    print("2D UMAP for visualization (n_neighbors=10, min_dist=0.1) saved.")
    print("Plotting UMAP 2D manifold...")
    plot_umap_2d(vis2d.embedding, loader.encoded_labels, loader.label_encoder, out_path=os.path.join(out_dir, "gtzan_umap_2d.png"))

    vis3d = GTZANManifold(n_components=3, model_name="gtzan_umap_vis_3d", 
                          n_neighbors=10, min_dist=0.1)
    vis3d.fit()
    vis3d.save()

    print("3D UMAP for visualization (n_neighbors=10, min_dist=0.1) saved.")
    print("Plotting and animating UMAP 3D manifold...")

    plot_umap_3d(
        vis3d.embedding, loader.encoded_labels, loader.label_encoder,
        interactive=True,
        out_path=os.path.join(out_dir, "gtzan_umap_3d.png"),
        animate_path=os.path.join(out_dir, "gtzan_umap_3d.gif")
    )

    print("All files saved in:", out_dir)
