# rec-engine/evaluation.py (DEFINITIVE THESIS VERSION with COLLABORATIVE FILTERING)

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- Import project modules ---
from utils import load_all_artifacts
from model import GatedMultimodalRecommender

# --- Import SOTA and CF components ---
try:
    from sentence_transformers.cross_encoder import CrossEncoder
    SOTA_TEXT_ENABLED = True
except ImportError:
    SOTA_TEXT_ENABLED = False

try:
    from surprise import Dataset, Reader, KNNWithMeans
    from surprise.model_selection import train_test_split
    CF_ENABLED = True
except ImportError:
    CF_ENABLED = False
    print("[WARNING] `scikit-surprise` not found. Collaborative Filtering evaluation will be skipped.")
    print("          To enable, run: pip install scikit-surprise")

# --- Configuration ---
EVALUATION_CONFIG = {
    "K": 10,
    "SAMPLE_SIZE": 100,
    "SOTA_CROSS_ENCODER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "NUM_SIMULATED_USERS": 500,
    "LIKES_PER_USER": 40
}

# --- All metric calculation helpers remain the same ---
def get_ground_truth(seed_song_idx, df):
    # ... (code is identical to previous version)
    seed_genre = df.iloc[seed_song_idx]['main_genre']
    ground_truth_indices = set(df[df['main_genre'] == seed_genre].index)
    ground_truth_indices.remove(seed_song_idx)
    return ground_truth_indices

def precision_at_k(recommended_indices, ground_truth_indices): # ...
    k = len(recommended_indices)
    if k == 0: return 0.0
    relevant_at_k = len(set(recommended_indices) & ground_truth_indices)
    return relevant_at_k / k

def recall_at_k(recommended_indices, ground_truth_indices): # ...
    if not ground_truth_indices: return 0.0
    relevant_at_k = len(set(recommended_indices) & ground_truth_indices)
    return relevant_at_k / len(ground_truth_indices)

def average_precision_at_k(recommended_indices, ground_truth_indices): # ...
    k = len(recommended_indices)
    if not ground_truth_indices or k == 0: return 0.0
    hits = 0
    score = 0.0
    for i, p in enumerate(recommended_indices):
        if p in ground_truth_indices:
            hits += 1
            score += hits / (i + 1.0)
    return score / min(len(ground_truth_indices), k)

def diversity_at_k(recommended_indices, df): # ...
    if len(recommended_indices) < 2: return 0.0
    rec_vectors = np.stack(df.loc[recommended_indices]['audio_features_vector'].values)
    sim_matrix = cosine_similarity(rec_vectors)
    upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
    avg_similarity = np.mean(sim_matrix[upper_triangle_indices])
    return 1 - avg_similarity

# --- All baseline functions remain the same ---
def get_random_recommendations(seed_idx, df, k): # ...
    return df.index.drop(seed_idx).to_series().sample(n=k, replace=False).tolist()

def get_simple_audio_recommendations(seed_idx, df, k): # ...
    source_vector = np.array([df.iloc[seed_idx]['audio_features_vector']])
    target_vectors = np.stack(df['audio_features_vector'].values)
    similarities = cosine_similarity(source_vector, target_vectors).flatten()
    return np.argsort(similarities)[-k-1:][::-1][1:].tolist()

def get_our_model_recommendations(seed_idx, df, model, mode, k): # ...
    seed_song = df.iloc[seed_idx]
    if mode == 'audio_focus':
        source_vec = torch.from_numpy(np.array([seed_song['audio_features_vector']])).float()
        source_emb = model.projection_activation(model.audio_projector(source_vec))
        target_vecs = torch.from_numpy(np.stack(df['audio_features_vector'].values)).float()
        target_embs = model.projection_activation(model.audio_projector(target_vecs))
    elif mode == 'lyric_focus':
        source_vec = torch.from_numpy(np.array([seed_song['text_features_vector']])).float()
        source_emb = model.projection_activation(model.text_projector(source_vec))
        target_vecs = torch.from_numpy(np.stack(df['text_features_vector'].values)).float()
        target_embs = model.projection_activation(model.text_projector(target_vecs))
    else:
        source_emb = torch.from_numpy(np.array([seed_song['final_embedding']])).float()
        target_embs = torch.from_numpy(np.stack(df['final_embedding'].values)).float()
    similarities = cosine_similarity(source_emb.detach().numpy(), target_embs.detach().numpy()).flatten()
    return np.argsort(similarities)[-k-1:][::-1][1:].tolist()

def get_sota_text_recommendations(seed_idx, df, cross_encoder_model, k): # ...
    seed_lyrics = df.iloc[seed_idx]['lyrics'];
    if pd.isna(seed_lyrics): seed_lyrics = ""
    candidate_indices = df.index.drop(seed_idx).to_series().sample(n=100, replace=False).tolist()
    candidate_lyrics = df.loc[candidate_indices]['lyrics'].fillna('').tolist()
    sentence_pairs = [[seed_lyrics, cand_lyrics] for cand_lyrics in candidate_lyrics]
    scores = cross_encoder_model.predict(sentence_pairs, show_progress_bar=False)
    ranked_candidates = sorted(zip(scores, candidate_indices), key=lambda x: x[0], reverse=True)
    return [idx for score, idx in ranked_candidates[:k]]


# --- NEW COLLABORATIVE FILTERING FUNCTIONS ---
def simulate_user_item_data(df):
    """Simulates user-item interaction data based on song popularity."""
    print("Simulating user-item interaction data based on popularity...")
    n_users = EVALUATION_CONFIG['NUM_SIMULATED_USERS']
    n_items = len(df)
    likes_per_user = EVALUATION_CONFIG['LIKES_PER_USER']
    
    # Popularity scores as sampling probabilities
    # We add a small constant to avoid zero probability
    probabilities = df['popularity_score'] + 0.01
    probabilities /= probabilities.sum()
    
    ratings_data = {'userID': [], 'itemID': [], 'rating': []}
    
    for user_id in tqdm(range(n_users), desc="Simulating Users"):
        # Each user "likes" a sample of songs, biased by popularity
        liked_indices = np.random.choice(df.index, size=likes_per_user, replace=False, p=probabilities)
        for item_idx in liked_indices:
            ratings_data['userID'].append(user_id)
            ratings_data['itemID'].append(item_idx)
            ratings_data['rating'].append(1.0) # We'll use a binary 'liked' system
            
    return pd.DataFrame(ratings_data)

def get_cf_recommendations(seed_idx, cf_model, k):
    """Gets recommendations from a trained scikit-surprise CF model."""
    try:
        # Get the inner ID that surprise uses
        inner_id = cf_model.trainset.to_inner_iid(seed_idx)
        # Get the K nearest neighbors
        neighbors = cf_model.get_neighbors(inner_id, k=k)
        # Convert inner IDs back to raw DataFrame indices
        rec_indices = [cf_model.trainset.to_raw_iid(inner) for inner in neighbors]
        return rec_indices
    except (ValueError, KeyError):
        # This can happen if a song had no interactions in the training set
        return []

# --- MAIN EVALUATION SCRIPT ---
if __name__ == "__main__":
    print("--- Starting Research-Grade Model Evaluation ---")
    K = EVALUATION_CONFIG['K']
    
    # 1. Load data and our trained model
    print("Loading data and models...")
    df_prepped, model, _, _, _ = load_all_artifacts()

    # 2. Load/Train all baseline models
    sota_text_model = None
    if SOTA_TEXT_ENABLED:
        sota_text_model = CrossEncoder(EVALUATION_CONFIG['SOTA_CROSS_ENCODER_MODEL'])

    cf_model = None
    if CF_ENABLED:
        simulated_df = simulate_user_item_data(df_prepped)
        reader = Reader(rating_scale=(1, 1)) # Binary likes
        data = Dataset.load_from_df(simulated_df, reader)
        trainset = data.build_full_trainset()
        
        print("Training Item-Based Collaborative Filtering model...")
        sim_options = {'name': 'cosine', 'user_based': False}
        cf_model = KNNWithMeans(k=K, sim_options=sim_options, verbose=False)
        cf_model.fit(trainset)
        print("✅ CF model trained.")

    # 3. Initialize results dictionary
    models_to_evaluate = {
        "Random": [], "Simple Audio": [],
        "Our Model (Balanced)": [], "Our Model (Audio Focus)": [], "Our Model (Lyric Focus)": []
    }
    if SOTA_TEXT_ENABLED: models_to_evaluate["SOTA Text (Cross-Encoder)"] = []
    if CF_ENABLED: models_to_evaluate["Item-Based CF"] = []

    # 4. Run evaluation loop
    print(f"\nRunning evaluation on a sample of {EVALUATION_CONFIG['SAMPLE_SIZE']} songs...")
    test_indices = df_prepped.sample(n=EVALUATION_CONFIG['SAMPLE_SIZE'], random_state=42).index

    for seed_idx in tqdm(test_indices, desc="Evaluating Models"):
        ground_truth = get_ground_truth(seed_idx, df_prepped)
        if not ground_truth: continue
        
        recs = {}
        recs["Random"] = get_random_recommendations(seed_idx, df_prepped, K)
        recs["Simple Audio"] = get_simple_audio_recommendations(seed_idx, df_prepped, K)
        recs["Our Model (Balanced)"] = get_our_model_recommendations(seed_idx, df_prepped, model, 'balanced', K)
        recs["Our Model (Audio Focus)"] = get_our_model_recommendations(seed_idx, df_prepped, model, 'audio_focus', K)
        recs["Our Model (Lyric Focus)"] = get_our_model_recommendations(seed_idx, df_prepped, model, 'lyric_focus', K)
        if SOTA_TEXT_ENABLED:
            recs["SOTA Text (Cross-Encoder)"] = get_sota_text_recommendations(seed_idx, df_prepped, sota_text_model, K)
        if CF_ENABLED:
            recs["Item-Based CF"] = get_cf_recommendations(seed_idx, cf_model, K)

        for name, rec_indices in recs.items():
            precision = precision_at_k(rec_indices, ground_truth)
            recall = recall_at_k(rec_indices, ground_truth)
            avg_prec = average_precision_at_k(rec_indices, ground_truth)
            diversity = diversity_at_k(rec_indices, df_prepped)
            models_to_evaluate[name].append([precision, recall, avg_prec, diversity])
            
    # 5. Aggregate and print results
    print("\n\n--- FINAL EVALUATION RESULTS ---")
    print("-" * 80)
    print(f"{'Model':<28} | {'MAP@10':<10} | {'Precision@10':<15} | {'Recall@10':<12} | {'Diversity@10':<15}")
    print("-" * 80)

    # Reorder for better storytelling in the final table
    order = [
        "Random", "Simple Audio", "SOTA Text (Cross-Encoder)", "Item-Based CF",
        "Our Model (Audio Focus)", "Our Model (Lyric Focus)", "Our Model (Balanced)"
    ]
    for name in order:
        if name not in models_to_evaluate: continue
        scores = models_to_evaluate[name]
        if not scores: continue
        avg_scores = np.mean(scores, axis=0)
        map_val, prec_val, rec_val, div_val = avg_scores[2], avg_scores[0], avg_scores[1], avg_scores[3]
        print(f"{name:<28} | {map_val:<10.3f} | {prec_val:<15.3f} | {rec_val:<12.3f} | {div_val:<15.3f}")
    
    print("-" * 80)
    print("\nEvaluation complete. This table is ready for your thesis report.")