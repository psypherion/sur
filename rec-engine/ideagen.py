# rec-engine/ideator.py (VERSION 7 - FINAL with CROSS-GENRE EXPLORATION)

import os
import torch
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Optional

from sklearn.metrics.pairwise import cosine_similarity
from utils import load_all_artifacts

# --- Gemini API Setup (Used for theme summarization) ---
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_SUMMARIZER = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Gemini API configured for lyrical theme summarization.")
    else:
        GEMINI_SUMMARIZER = None
except (ImportError, TypeError):
    GEMINI_SUMMARIZER = None

GEMINI_THEME_PROMPT = """
<task_definition>
You are an expert musicologist and lyrical analyst, tasked with distilling the thematic essence of a song into a single, potent sentence. Your analysis should go beyond surface-level descriptions and capture the core emotional narrative, central conflict, or philosophical message of the lyrics.

Your output must be a single, concise sentence. Do not add any conversational text or introductions.
</task_definition>

<examples>
  <example>
    <input>
      <song title="Bohemian Rhapsody" artist="Queen" />
      <lyrics>
        Is this the real life? Is this just fantasy?
        Caught in a landslide, no escape from reality
        Open your eyes, look up to the skies and see
        I'm just a poor boy, I need no sympathy
        Because I'm easy come, easy go, little high, little low
        Any way the wind blows doesn't really matter to me, to me
        ...
        So you think you can stone me and spit in my eye?
        So you think you can love me and leave me to die?
        Oh, baby, can't do this to me, baby!
        Just gotta get out, just gotta get right outta here!
      </lyrics>
    </input>
    <output>
      A young man confesses a murder to his mother and confronts his nihilistic despair as he faces his own mortality.
    </output>
  </example>

  <example>
    <input>
      <song title="Losing My Religion" artist="R.E.M." />
      <lyrics>
        Oh, life is bigger
        It's bigger than you and you are not me
        The lengths that I will go to
        The distance in your eyes
        Oh no, I've said too much
        I set it up
        ...
        That's me in the corner
        That's me in the spotlight, I'm
        Losing my religion
        Trying to keep up with you
        And I don't know if I can do it
        Oh no, I've said too much
        I haven't said enough
      </lyrics>
    </input>
    <output>
      The narrator grapples with obsessive thoughts and the agonizing vulnerability of unrequited love, fearing they have revealed too much of their inner turmoil.
    </output>
  </example>
</examples>

<user_request>
  <input>
    <song title="{title}" artist="{artist}" />
    <lyrics>
      {lyrics}
    </lyrics>
  </input>
  <output>
  </output>
</user_request>

"""

# --- All helper and analysis functions from previous versions ---
# (These functions are reused and do not need changes)
def get_theme_summary_from_gemini(song: pd.Series) -> str:
    if not GEMINI_SUMMARIZER: return "Theme summarization is disabled (API key not found)."
    lyrics = song.get('lyrics', '')
    if not isinstance(lyrics, str) or not lyrics.strip(): return "Instrumental"
    prompt = GEMINI_THEME_PROMPT.format(title=song['title'], artist=song['artist'], lyrics=lyrics)
    try:
        response = GEMINI_SUMMARIZER.generate_content(prompt)
        return response.text.strip()
    except Exception: return "Could not generate theme summary."

def get_sonality_profile(neighborhood_df: pd.DataFrame) -> str:
    if neighborhood_df.empty: return "A truly unique and uncharted sonic territory."
    # ... (code is identical to previous version)
    energy = neighborhood_df['energy'].mean()
    valence = neighborhood_df['valence'].mean()
    acousticness = neighborhood_df['acousticness'].mean()
    personality_traits = []
    if energy > 0.7: personality_traits.append("Energetic")
    elif energy < 0.4: personality_traits.append("Calm / Mellow")
    else: personality_traits.append("Mid-Energy")
    if valence > 0.6: personality_traits.append("Upbeat & Happy")
    elif valence > 0.4: personality_traits.append("Neutral / Mixed-Mood")
    else: personality_traits.append("Melancholic & Somber")
    if acousticness > 0.7: personality_traits.append("Primarily Acoustic")
    elif acousticness < 0.1: personality_traits.append("Highly Electronic/Electric")
    top_genres = neighborhood_df['main_genre'].value_counts().nlargest(2).index.tolist()
    genre_str = " & ".join([g.capitalize() for g in top_genres])
    return f"A blend of {genre_str} styles, generally characterized as: {', '.join(personality_traits)}."


def vectorize_user_prompt(prompt: str, embedding_model, sentiment_pipeline) -> np.ndarray:
    print("   > Vectorizing your creative idea...")
    # ... (code is identical to previous version)
    if not prompt.strip():
        embedding_dim = embedding_model.get_sentence_embedding_dimension()
        return np.array([0.0] * (embedding_dim + 3))
    embedding = embedding_model.encode([prompt])
    semantic_embedding = embedding[0].tolist()
    sentiment_results = sentiment_pipeline(prompt, truncation=True)
    scores = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
    result = sentiment_results[0]
    label = result['label'].lower()
    scores[label] = result['score']
    sentiment_vector = [scores['negative'], scores['neutral'], scores['positive']]
    return np.array(semantic_embedding + sentiment_vector)

def analyze_search_results(prepped_results_df: pd.DataFrame, raw_data_df: pd.DataFrame):
    if prepped_results_df.empty:
        print("\nCould not find any songs matching your criteria within the selected genre.")
        return
    # ... (code is identical to previous version)
    raw_results_df = raw_data_df.loc[prepped_results_df.index]
    print("\n" + "="*65)
    print("💡               CREATIVE IDEATION REPORT               💡")
    print("="*65)
    print("\nBased on your idea, here is an analysis of the most lyrically similar songs:")
    sonality = get_sonality_profile(raw_results_df)
    print(f"\nSonality™ Profile: {sonality}")
    print("-" * 40)
    all_progs = [p['main'] for p in raw_results_df['chord_progression'] if isinstance(p, dict) and 'main' in p and p['main']]
    if all_progs:
        prog_counts = Counter(all_progs).most_common(3)
        print("\n🎶 Common & Successful Chord Progressions:")
        for i, (prog, count) in enumerate(prog_counts):
            print(f"   {i+1}. {prog} (found in {count} similar song(s))")
    most_popular = raw_results_df.sort_values('popularity', ascending=False).head(3)
    print("\n🔥 To understand what resonates with listeners, check out:")
    for _, row in most_popular.iterrows():
        print(f"   - '{row['title']}' by {row['artist']} (Popularity: {int(row['popularity'])})")
    most_lyrical = prepped_results_df.sort_values('alpha', ascending=True).iloc[0]
    most_audio = prepped_results_df.sort_values('alpha', ascending=False).iloc[0]
    print("\n⚖️ For creative direction, consider the audio/lyric balance:")
    print(f"   - For LYRICAL inspiration: '{most_lyrical['title']}' (α = {most_lyrical['alpha']:.2f})")
    print(f"   - For SONIC/INSTRUMENTAL inspiration: '{most_audio['title']}' (α = {most_audio['alpha']:.2f})")
    print("="*65)


# --- NEW FUNCTION AND REPORT FOR CROSS-GENRE MODE ---
def analyze_cross_genre_results(source_songs: pd.DataFrame, cross_genre_results: pd.DataFrame):
    """Generates a special report for the cross-genre inspiration mode."""
    print("\n" + "="*65)
    print("💡         CROSS-GENRE THEMATIC EXPLORER          💡")
    print("="*65)

    print("\nSTEP 1: IDENTIFYING THE CORE THEME")
    print("---------------------------------------")
    print("The core lyrical theme was derived from these popular source songs:")
    for _, song in source_songs.iterrows():
        print(f"  - '{song['title']}' by {song['artist']}")

    print("\nSTEP 2: FINDING THE THEME IN OTHER GENRES")
    print("---------------------------------------")
    if cross_genre_results.empty:
        print("Could not find this theme in any other genres. It might be unique!")
    else:
        print("Here is how that same lyrical theme appears across different musical styles:")
        # Group results by genre for a clean, organized report
        for genre, group in cross_genre_results.groupby('main_genre'):
            print(f"\n  In {genre.upper()}:")
            for _, song in group.head(2).iterrows(): # Show top 2 from each genre
                print(f"    - '{song['title']}' by {song['artist']}")
    
    print("\nListen to these songs to get inspired by how a single idea can be expressed in many ways.")
    print("="*65)


# --- MAIN APPLICATION BLOCK (UPDATED WITH THREE MODES) ---
if __name__ == "__main__":
    print("Loading song database and models...")
    from preprocessor import load_and_consolidate_data
    raw_song_df = pd.DataFrame(load_and_consolidate_data("db"))
    df_prepped, model, scalar_scaler, chord_map, kmeans_model = load_all_artifacts()
    raw_song_df = raw_song_df.loc[df_prepped.index]
    print(f"✅ Database loaded with {len(df_prepped)} songs.")

    from sentence_transformers import SentenceTransformer
    from transformers import pipeline, logging as hf_logging
    hf_logging.set_verbosity_error()
    print("Initializing NLP models for query vectorization...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model_instance = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
    sentiment_pipeline_instance = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment", device=0 if device == "cuda" else -1)
    print("✅ NLP models ready.")

    AVAILABLE_GENRES = sorted(df_prepped['main_genre'].unique().tolist())
    print("\n\nWelcome to the Musician's Idea Generator!")

    while True:
        try:
            # STEP 1: GENRE SELECTION
            print("\n-------------------------------------------")
            print("STEP 1: Please select a genre.")
            for i, genre in enumerate(AVAILABLE_GENRES):
                print(f"   {i+1}. {genre}")
            print("   0. Exit")
            choice = input("Enter the number for your choice > ")
            if not choice.isdigit() or not (0 <= int(choice) <= len(AVAILABLE_GENRES)):
                print("Invalid input. Please enter a number from the list.")
                continue
            choice_num = int(choice)
            if choice_num == 0: break
            target_genre = AVAILABLE_GENRES[choice_num - 1]

            # STEP 2: MODE SELECTION
            print("\nSTEP 2: What do you want to do?")
            print("   1. Explore a lyrical idea I already have (Search Mode).")
            print("   2. Get popular theme summaries for this genre (Inspiration Mode).")
            print("   3. Find how this genre's themes appear elsewhere (Cross-Genre Explorer).")
            mode_choice = input("Enter 1, 2, or 3 > ")

            if mode_choice == '1': # --- SEARCH MODE ---
                print("\nSTEP 3: Describe the lyrical theme or mood of your song.")
                user_prompt = input("Describe your idea > ")
                if not user_prompt.strip(): continue
                search_df = df_prepped[df_prepped['main_genre'] == target_genre]
                query_vector = vectorize_user_prompt(user_prompt, embedding_model_instance, sentiment_pipeline_instance)
                db_text_vectors = np.stack(search_df['text_features_vector'].values)
                similarities = cosine_similarity(query_vector.reshape(1, -1), db_text_vectors).flatten()
                top_results_df = search_df.iloc[np.argsort(similarities)[-15:][::-1]]
                analyze_search_results(top_results_df, raw_song_df)

            elif mode_choice == '2': # --- INSPIRATION MODE ---
                print(f"\nAnalyzing popular lyrical themes in '{target_genre}'...")
                search_df = raw_song_df[raw_song_df['main_genre'] == target_genre]
                popular_songs = search_df.sort_values('popularity', ascending=False).head(3)
                # ... (code is identical to previous version) ...
                if popular_songs.empty:
                    print("Could not find enough popular songs in this genre for analysis.")
                    continue
                print("\n" + "="*65)
                print("💡            LYRICAL THEME INSPIRATION             💡")
                print("="*65)
                print("\nHere are the themes of the most popular songs in this genre:")
                for _, song in popular_songs.iterrows():
                    print(f"\n  - '{song['title']}' by {song['artist']} (Popularity: {int(song['popularity'])})")
                    theme = get_theme_summary_from_gemini(song)
                    print(f"    Theme: {theme}")
                print("\nConsider writing about similar topics to appeal to this audience.")
                print("="*65)

            elif mode_choice == '3': # --- CROSS-GENRE EXPLORER MODE ---
                print(f"\nCreating a thematic profile for '{target_genre}' to search other genres...")
                
                # 1. Find the most popular songs in the source genre
                source_songs_df = df_prepped[df_prepped['main_genre'] == target_genre]
                if len(source_songs_df) < 3:
                    print("Not enough songs in this genre to create a reliable theme profile.")
                    continue
                
                popular_source_songs = source_songs_df.sort_values('popularity', ascending=False).head(3)
                
                # 2. Create the average "theme vector" from these popular songs
                theme_vectors = np.stack(popular_source_songs['text_features_vector'].values)
                avg_theme_vector = np.mean(theme_vectors, axis=0).reshape(1, -1)
                
                # 3. Search the ENTIRE database with this theme vector
                all_db_text_vectors = np.stack(df_prepped['text_features_vector'].values)
                similarities = cosine_similarity(avg_theme_vector, all_db_text_vectors).flatten()
                
                # 4. Get top results and filter out the source genre
                top_indices = np.argsort(similarities)[-20:][::-1] # Get more results initially
                top_results_df = df_prepped.iloc[top_indices]
                cross_genre_results = top_results_df[top_results_df['main_genre'] != target_genre].head(5)
                
                # 5. Generate the special report
                analyze_cross_genre_results(popular_source_songs, cross_genre_results)

            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

        except KeyboardInterrupt: break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()

    print("\nHappy creating! Exiting.")