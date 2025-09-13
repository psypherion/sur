import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import joblib
import numpy as np
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()

class _GTZANLoader:
    def __init__(self) -> None:
        self.GTZAN_CSV = os.getenv("GTZAN_CSV")
        if not self.GTZAN_CSV:
            raise ValueError("GTZAN_CSV path not found in environment variables.")
        self.gtzan_csv = os.path.join(os.getcwd(), self.GTZAN_CSV)
        if not os.path.isfile(self.gtzan_csv):
            raise FileNotFoundError(f"GTZAN CSV file not found at path: {self.gtzan_csv}")
        # Show progress loading CSV
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
        # TQDM here not really needed as scaler is vectorized, but for feedback:
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)

class GTZANManifold:
    def __init__(self, loader: _GTZANLoader, n_components: int, model_name: str, 
                 n_neighbors=15, min_dist=0.1):
        self.loader = loader
        self.n_components = n_components
        self.model_name = model_name
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.umap_model = None
        self.embedding = None

    def fit(self):
        print(f"Fitting UMAP ({self.n_components}D) manifold [n_neighbors={self.n_neighbors}, min_dist={self.min_dist}]: this may take a few moments...")
        with tqdm(total=1, desc=f"UMAP {self.n_components}D") as pbar:
            self.umap_model = umap.UMAP(
                n_components=self.n_components, 
                random_state=42, 
                n_neighbors=self.n_neighbors, 
                min_dist=self.min_dist
            )
            self.embedding = self.umap_model.fit_transform(self.loader.scaled_features)
            pbar.update(1)
    
    def save(self, out_dir="./audio_manifold/"):
        os.makedirs(out_dir, exist_ok=True)
        # Save UMAP model
        joblib.dump(self.umap_model, os.path.join(out_dir, f"{self.model_name}.pkl"))
        # Save embedding as CSV
        col_names = [f"UMAP_{i+1}" for i in range(self.n_components)]
        emb_df = pd.DataFrame(np.array(self.embedding), columns=col_names)
        emb_df['label_id'] = self.loader.encoded_labels
        emb_df['label'] = self.loader.labels.values
        emb_df.to_csv(os.path.join(out_dir, f"{self.model_name}_embedding.csv"), index=False)

def plot_umap_2d(embedding, labels, label_encoder, out_path=None):
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

if __name__ == "__main__":
    out_dir = "./audio_manifold/"

    # --- 1. Load data ---
    print("Preparing GTZAN data...")
    loader = _GTZANLoader()

    # --- 2. Build and save 2D UMAP for visualization (try sharp clusters)
    vis_manifold = GTZANManifold(loader, n_components=2, model_name="gtzan_umap_vis_2d",
                                n_neighbors=10, min_dist=0.05)
    vis_manifold.fit()
    vis_manifold.save(out_dir=out_dir)
    print("2D UMAP for visualization (n_neighbors=10, min_dist=0.05) saved.")

    # --- 3. Try alternative (global) for comparison
    vis_manifold_global = GTZANManifold(loader, n_components=2, model_name="gtzan_umap_vis_2d_global",
                                        n_neighbors=50, min_dist=0.5)
    vis_manifold_global.fit()
    vis_manifold_global.save(out_dir=out_dir)
    print("2D UMAP (n_neighbors=50, min_dist=0.5) saved.")

    # --- 3. Build and save high-dim UMAP for ML ---
    ml_dim = 32
    ml_manifold = GTZANManifold(loader, n_components=ml_dim, model_name=f"gtzan_umap_ml_{ml_dim}d")
    ml_manifold.fit()
    ml_manifold.save(out_dir=out_dir)
    print(f"{ml_dim}D UMAP for ML saved.")

    # --- 4. Save scaler and label encoder ---
    print("Saving scaler and label encoder...")
    joblib.dump(loader.scaler, os.path.join(out_dir, "gtzan_scaler.pkl"))
    joblib.dump(loader.label_encoder, os.path.join(out_dir, "gtzan_label_encoder.pkl"))

    # --- 5. Show UMAP 2D plot ---
    print("Plotting UMAP 2D manifold...")
    plot_umap_2d(vis_manifold.embedding, loader.encoded_labels, loader.label_encoder,
                 out_path=os.path.join(out_dir, "gtzan_umap_2d.png"))

    print("All files saved in:", out_dir)
