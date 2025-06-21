# rec-engine/train.py

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple

from model import GatedMultimodalRecommender

# --- Configuration ---
PROCESSED_DATA_PATH = "processed_data/preprocessed_song_data.json"
MODEL_SAVE_PATH = "gated_recommender_model.pth"

# Using hyperparameters that are proven to work well for this simpler task
CONFIG = {
    "optimizer_choice": "AdamW",
    "epochs": 500,  # 50 epochs is more than enough for this clearer task
    "batch_size": 32,
    "learning_rate": 0.0007, # Can use a slightly higher LR for faster convergence
    "embedding_dim": 256,
    "triplet_margin": 0.2
}

# --- Simplified and Robust Dataset Class ---
class SongSimilarityDataset(Dataset):
    """
    Generates (anchor, positive, negative) triplets based on main genre.
    - Positive: A different song from the same main genre.
    - Negative: A song from a different main genre.
    This provides a clear and strong signal for training a similarity model.
    """
    def __init__(self, data: List[Dict]):
        self.data = data
        self.df = pd.DataFrame(data).reset_index().rename(columns={'index': 'id'})
        self.genre_map = self.df.groupby('main_genre')['id'].apply(list)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple:
        anchor_song = self.df.iloc[index]
        anchor_genre = anchor_song['main_genre']

        # Get a positive sample (same main genre)
        possible_positives = [i for i in self.genre_map[anchor_genre] if i != index]
        positive_index = random.choice(possible_positives if possible_positives else [index])

        # Get a negative sample (different main genre)
        negative_genre = random.choice([g for g in self.genre_map.keys() if g != anchor_genre])
        negative_index = random.choice(self.genre_map[negative_genre])

        # Retrieve data
        anchor, positive, negative = self.data[index], self.data[positive_index], self.data[negative_index]
        
        # Convert to tensors
        return (torch.tensor(anchor['audio_features_vector'], dtype=torch.float32), torch.tensor(anchor['text_features_vector'], dtype=torch.float32)), \
               (torch.tensor(positive['audio_features_vector'], dtype=torch.float32), torch.tensor(positive['text_features_vector'], dtype=torch.float32)), \
               (torch.tensor(negative['audio_features_vector'], dtype=torch.float32), torch.tensor(negative['text_features_vector'], dtype=torch.float32))

# --- Main Training Script ---
if __name__ == "__main__":
    with open(PROCESSED_DATA_PATH, 'r') as f: data = json.load(f)

    AUDIO_DIM = len(data[0]['audio_features_vector'])
    TEXT_DIM = len(data[0]['text_features_vector'])
    
    print("--- Simplified (Main Genre) Training Pipeline ---")
    print(f"Optimizer: {CONFIG['optimizer_choice']}")
    # ... print other configs ...

    dataset = SongSimilarityDataset(data)
    # Using num_workers can speed up data loading
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    
    model = GatedMultimodalRecommender(
        audio_input_dim=AUDIO_DIM, text_input_dim=TEXT_DIM, embedding_dim=CONFIG['embedding_dim']
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)
    loss_fn = nn.TripletMarginLoss(margin=CONFIG['triplet_margin'], p=2)
    cos_sim = nn.CosineSimilarity(dim=1)

    print("\nStarting model training...")
    model.train()

    for epoch in range(CONFIG['epochs']):
        total_loss, total_pos_sim, total_neg_sim = 0, 0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch in pbar:
            (a_audio, a_text), (p_audio, p_text), (n_audio, n_text) = batch
            optimizer.zero_grad()
            
            a_emb, _ = model(a_audio, a_text)
            p_emb, _ = model(p_audio, p_text)
            n_emb, _ = model(n_audio, n_text)
            
            loss = loss_fn(a_emb, p_emb, n_emb)
            loss.backward()
            optimizer.step()

            # Monitoring
            with torch.no_grad():
                pos_sim = cos_sim(a_emb, p_emb).mean().item()
                neg_sim = cos_sim(a_emb, n_emb).mean().item()
            total_loss += loss.item()
            total_pos_sim += pos_sim
            total_neg_sim += neg_sim
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Pos_Sim': f"{pos_sim:.2f}", 'Neg_Sim': f"{neg_sim:.2f}"})

        avg_loss = total_loss / len(dataloader)
        avg_pos_sim = total_pos_sim / len(dataloader)
        avg_neg_sim = total_neg_sim / len(dataloader)

        print(
            f"Epoch {epoch+1} Summary -> Avg Loss: {avg_loss:.4f} | "
            f"Avg Cosine Sim -> Pos: {avg_pos_sim:.2f}, Neg: {avg_neg_sim:.2f}"
        )
        scheduler.step(avg_loss)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Training complete. Model saved to '{MODEL_SAVE_PATH}'")