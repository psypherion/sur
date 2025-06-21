# rec-engine/recommender.py (DEFINITIVE FINAL VERSION)

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional, Tuple

# Import our project's modules
from utils import load_all_artifacts
from model import GatedMultimodalRecommender

@torch.no_grad()
def find_similar_songs(
    seed_song: pd.Series,
    df_prepped: pd.DataFrame,
    model: GatedMultimodalRecommender,
    mode: str = 'balanced',
    top_n: int = 5
) -> pd.DataFrame:
    """
    Finds and ranks similar songs based on different modes of comparison.
    This version is optimized to prevent PyTorch UserWarnings.
    """
    model.eval()
    
    # 1. Get the source vector and target vectors based on the chosen mode
    if mode == 'audio_focus':
        # Project the raw audio features into the learned latent space
        source_audio_vec = torch.from_numpy(np.array([seed_song['audio_features_vector']])).float()
        source_vector = model.projection_activation(model.audio_projector(source_audio_vec))
        # Compare against all other songs' projected audio vectors
        target_vectors_np = np.stack(df_prepped['audio_features_vector'].values)
        target_vectors = model.projection_activation(model.audio_projector(torch.from_numpy(target_vectors_np).float()))
        
    elif mode == 'lyric_focus':
        # Project the raw text features into the learned latent space
        source_text_vec = torch.from_numpy(np.array([seed_song['text_features_vector']])).float()
        source_vector = model.projection_activation(model.text_projector(source_text_vec))
        # Compare against all other songs' projected text vectors
        target_vectors_np = np.stack(df_prepped['text_features_vector'].values)
        target_vectors = model.projection_activation(model.text_projector(torch.from_numpy(target_vectors_np).float()))
        
    else: # 'balanced' mode (default)
        # Use the final, gated embedding that the model learned
        source_vector = torch.from_numpy(np.array([seed_song['final_embedding']])).float()
        target_vectors_np = np.stack(df_prepped['final_embedding'].values)
        target_vectors = torch.from_numpy(target_vectors_np).float()

    # 2. Calculate cosine similarity between the source vector and all target vectors
    similarities = cosine_similarity(source_vector.numpy(), target_vectors.numpy()).flatten()
    
    # 3. Rank and format the results
    df_prepped['similarity'] = similarities
    # Sort by similarity and remove the seed song itself from the recommendations
    recommendations = df_prepped[df_prepped.index != seed_song.name].sort_values(
        by='similarity', ascending=False
    ).head(top_n)
    
    return recommendations

def explain_recommendation(seed_song: pd.Series, recommended_song: pd.Series, mode: str) -> str:
    """
    Generates a dynamic, human-readable explanation for a single recommendation
    that adapts to the recommendation mode and similarity scores.
    """
    overall_sim = recommended_song['similarity']
    
    # Calculate raw content similarities for deeper explanation
    audio_sim = cosine_similarity(
        [seed_song['audio_features_vector']],
        [recommended_song['audio_features_vector']]
    )[0][0]
    
    text_sim = cosine_similarity(
        [seed_song['text_features_vector']],
        [recommended_song['text_features_vector']]
    )[0][0]
    
    # Dynamically generate the core reason based on the mode and scores
    explanation_intro = f"  - Overall Match Score: {overall_sim:.0%}\n"
    
    if mode == 'audio_focus' or (mode == 'balanced' and audio_sim > text_sim + 0.1):
        reason = f"    - Reason: This song is recommended primarily for its **sonic similarity** ({audio_sim:.0%})."
    elif mode == 'lyric_focus' or (mode == 'balanced' and text_sim > audio_sim + 0.1):
        reason = f"    - Reason: This song is recommended primarily for its **lyrical theme** ({text_sim:.0%})."
    else:
        reason = f"    - Reason: This is a strong **balanced match**, with both sonic ({audio_sim:.0%}) and lyrical ({text_sim:.0%}) elements contributing."
        
    # Add a final insight about the alpha values
    seed_alpha = seed_song['alpha']
    insight = f"\n    - Insight: Your song is {seed_alpha:.0%} audio-driven, showing how its identity is balanced between sound and lyrics."

    return explanation_intro + reason + insight

if __name__ == "__main__":
    print("Loading recommendation engine...")
    df_prepped, model, _, _, _ = load_all_artifacts()
    # Also load raw data for display purposes
    from preprocessor import load_and_consolidate_data
    raw_song_df = pd.DataFrame(load_and_consolidate_data("db"))
    raw_song_df = raw_song_df.loc[df_prepped.index]

    print("✅ Recommendation Engine Ready.\n")
    
    while True:
        try:
            # --- 1. Get Seed Song ---
            song_query = input("Enter a song title to get recommendations for (or 'exit') > ")
            if song_query.lower() == 'exit': break
            
            # Find the song (case-insensitive search)
            matches = df_prepped[df_prepped['title'].str.contains(song_query, case=False, regex=False)]
            
            if matches.empty:
                print(f"Sorry, couldn't find any song matching '{song_query}'. Please try again.")
                continue
            
            # Handle multiple matches by asking the user to choose
            if len(matches) > 1:
                print("Found multiple matches. Please choose one:")
                for i, row in matches.iterrows():
                    print(f"  {i+1}. '{row['title']}' by {row['artist']}")
                choice = input(f"Enter a number (1-{len(matches)}) > ")
                if choice.isdigit() and 1 <= int(choice) <= len(matches):
                    seed_song_prepped = matches.iloc[int(choice) - 1]
                else:
                    print("Invalid choice. Please try again.")
                    continue
            else:
                seed_song_prepped = matches.iloc[0]

            print(f"Found seed song: '{seed_song_prepped['title']}' by {seed_song_prepped['artist']}")

            # --- 2. Get Recommendation Mode ---
            print("\nSelect a recommendation mode:")
            print("  1. Balanced (Default): Best overall match using the model's learned logic.")
            print("  2. Audio Focus: Prioritize songs that SOUND similar.")
            print("  3. Lyrical Focus: Prioritize songs with a similar THEME.")
            mode_choice = input("Enter your choice (1, 2, or 3) > ")
            
            mode_map = {'1': 'balanced', '2': 'audio_focus', '3': 'lyric_focus'}
            mode = mode_map.get(mode_choice, 'balanced')
            print(f"\n--- Generating recommendations with '{mode.replace('_', ' ').capitalize()}' ---\n")

            # --- 3. Find and Display Recommendations ---
            recommendations = find_similar_songs(seed_song_prepped, df_prepped, model, mode=mode, top_n=5)

            if recommendations.empty:
                print("Could not generate recommendations for this song.")
                continue

            print(f"Top 5 Recommendations for '{seed_song_prepped['title']}':\n")
            for _, rec_song in recommendations.iterrows():
                print(f"🎶 '{rec_song['title']}' by {rec_song['artist']} ({rec_song['main_genre']})")
                # Pass the 'mode' to the explanation function for dynamic text
                explanation = explain_recommendation(seed_song_prepped, rec_song, mode)
                print(explanation)
                print("-" * 50)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()

    print("\nExiting recommender.")