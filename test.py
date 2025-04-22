import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import seaborn as sns
import pandas as pd

# Sample lyrics (you can replace these with your dataset)
lyrics = [
    "I loved her in the silence of the stars",
    "Baby, I miss you every night",
    "She runs through the fields like freedom itself",
    "Money, cars, fame—it’s all the same",
    "The algorithm chose love, not me",
    "We danced like time forgot to count",
]

# Optional genres/labels for coloring the plot
genres = ["Love", "Love", "Freedom", "Materialism", "Tech", "Nostalgia"]

# 1. Sentence-BERT Embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(lyrics)

# 2. Dimensionality Reduction (PCA and t-SNE)
reduced_pca = PCA(n_components=2).fit_transform(embeddings)
reduced_tsne = TSNE(n_components=2, random_state=42, perplexity=2).fit_transform(embeddings)

# 3. Plotting in subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# PCA plot
df_pca = pd.DataFrame(reduced_pca, columns=["x", "y"])
df_pca["genre"] = genres
sns.scatterplot(data=df_pca, x="x", y="y", hue="genre", s=100, palette="viridis", ax=axes[0])
axes[0].set_title("PCA of Sentence-BERT Lyric Embeddings")
axes[0].set_xlabel("Component 1")
axes[0].set_ylabel("Component 2")
axes[0].grid(True)

# t-SNE plot
df_tsne = pd.DataFrame(reduced_tsne, columns=["x", "y"])
df_tsne["genre"] = genres
sns.scatterplot(data=df_tsne, x="x", y="y", hue="genre", s=100, palette="viridis", ax=axes[1])
axes[1].set_title("t-SNE of Sentence-BERT Lyric Embeddings")
axes[1].set_xlabel("Component 1")
axes[1].set_ylabel("Component 2")
axes[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
