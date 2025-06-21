# rec-engine/preprocessor.py (DEFINITIVE FINAL VERSION)

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from transformers import pipeline, logging as hf_logging
import torch
import joblib
import librosa
from pydub import AudioSegment
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

# --- Configuration Block ---
CONFIG = {
    "source_dir": "db",
    "output_dir": "processed_data",
    "output_filename": "preprocessed_song_data.json",
    "scalar_scaler_filename": "scalar_scaler.pkl",
    "mfcc_scaler_filename": "mfcc_scaler.pkl",
    "chord_map_filename": "chord_map.json",
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "sentiment_model": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "scalar_audio_features": [
        'bpm', 'danceability', 'energy', 'acousticness', 'instrumentalness',
        'liveness', 'loudness', 'speechiness', 'valence'
    ],
    "processing_batch_size": 32,
    "n_mfcc": 13,
    "preview_clip_duration_ms": 10000
}

PROCESS_AUDIO_FILES = False
hf_logging.set_verbosity_error()

embedding_model = None
sentiment_pipeline = None

def initialize_text_models():
    """Initializes and loads the language models only when needed."""
    global embedding_model, sentiment_pipeline
    if embedding_model is None:
        print("Initializing text processing models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device.upper()}")
        embedding_model = SentenceTransformer(CONFIG["embedding_model"], device=device)
        sentiment_pipeline = pipeline("sentiment-analysis", model=CONFIG["sentiment_model"], tokenizer=CONFIG["sentiment_model"], device=0 if device == "cuda" else -1)
        print("Text models initialized.")

def load_and_consolidate_data(source_dir: str) -> List[Dict[str, Any]]:
    """
    Loads, consolidates, and de-duplicates all source JSON files.
    This version correctly prioritizes the 'genre' field within each song's data.
    """
    all_songs = []
    genre_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    print(f"\nFound {len(genre_files)} genre files to process in '{source_dir}'")
    
    for filename in tqdm(genre_files, desc="Consolidating JSON files"):
        fallback_genre = os.path.splitext(filename)[0].replace('-cp', '')
        filepath = os.path.join(source_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for song in data:
                    genre_from_song = song.get('genre')
                    if isinstance(genre_from_song, str) and genre_from_song.strip():
                        final_genre = genre_from_song
                    else:
                        final_genre = fallback_genre
                    song['main_genre'] = final_genre.capitalize()
                all_songs.extend(data)
        except (IOError, json.JSONDecodeError) as e:
            print(f"  [WARNING] Could not read/parse {filename}. Skipping. Error: {e}")
            
    df = pd.DataFrame(all_songs)
    df.drop_duplicates(subset=['title', 'artist'], keep='first', inplace=True)
    print(f"Consolidated and de-duplicated to a total of {len(df)} unique songs.")
    return df.to_dict('records')

def process_metadata_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler, Dict]:
    """Processes features from JSON metadata: scalars and chords."""
    print("\nProcessing Metadata Features (Scalars and Chords)...")
    df_processed = df.copy()
    scaler = MinMaxScaler()
    existing_scalars = [f for f in CONFIG["scalar_audio_features"] if f in df_processed.columns]
    df_processed[existing_scalars] = scaler.fit_transform(df_processed[existing_scalars])
    print(f"  > Normalized {len(existing_scalars)} scalar features.")
    df_processed['chord_progression'] = df_processed['chord_progression'].apply(lambda x: x if isinstance(x, dict) else {})
    all_chords = set(chord for prog in df_processed['chord_progression'] for sect in prog.values() for chord in str(sect).split(' - '))
    unique_chords = sorted(list(all_chords))
    chord_map = {chord: i for i, chord in enumerate(unique_chords)}
    harmonic_vectors = []
    for prog in df_processed['chord_progression']:
        vec = np.zeros(len(unique_chords), dtype=int)
        for sect in prog.values():
            for chord in str(sect).split(' - '):
                if chord in chord_map:
                    vec[chord_map[chord]] = 1
        harmonic_vectors.append(vec.tolist())
    df_processed['harmonic_vector'] = harmonic_vectors
    print(f"  > Created harmonic vectors with a vocabulary of {len(unique_chords)} chords.")
    return df_processed, scaler, chord_map

def get_audio_file_features(filepath: str) -> Optional[np.ndarray]:
    """Extracts MFCCs from a 30-second preview of an audio file."""
    try:
        audio = AudioSegment.from_mp3(filepath)
        duration_ms = len(audio)
        clip_len = CONFIG["preview_clip_duration_ms"]
        if duration_ms < clip_len * 3:
             preview = audio
        else:
            start_clip = audio[:clip_len]
            middle_start = (duration_ms - clip_len) // 2
            middle_clip = audio[middle_start : middle_start + clip_len]
            end_clip = audio[-clip_len:]
            preview = start_clip + middle_clip + end_clip
        samples = np.array(preview.get_array_of_samples()).astype(np.float32)
        sr = preview.frame_rate
        if preview.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        y = librosa.util.buf_to_float(samples)
        # --- FINAL FIX IS HERE: Use n_mfcc instead of n_cc ---
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG["n_mfcc"])
        return np.mean(mfccs, axis=1)
    except Exception:
        return None

def process_audio_files(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[MinMaxScaler]]:
    """Orchestrates the extraction and normalization of MFCCs from audio files."""
    print("\nProcessing Audio Files to Extract MFCCs...")
    mfcc_vectors = [get_audio_file_features(fp) for fp in tqdm(df['audio_file_path'].fillna(''), desc="Extracting MFCCs")]
    valid_mfccs = [v for v in mfcc_vectors if v is not None]
    if not valid_mfccs:
        print("  > No valid audio files found or processed. MFCC vectors will be all zeros.")
        df['mfcc_vector'] = [[0.0] * CONFIG["n_mfcc"]] * len(df)
        return df, None
    mfcc_scaler = MinMaxScaler()
    scaled_valid_mfccs = mfcc_scaler.fit_transform(valid_mfccs)
    scaled_mfcc_list, mfcc_iter = [], iter(scaled_valid_mfccs)
    for vec in mfcc_vectors:
        if vec is not None:
            scaled_mfcc_list.append(next(mfcc_iter).tolist())
        else:
            scaled_mfcc_list.append([0.0] * CONFIG["n_mfcc"])
    df['mfcc_vector'] = scaled_mfcc_list
    print(f"  > Generated and normalized MFCC vectors for {len(valid_mfccs)} songs.")
    return df, mfcc_scaler

def process_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates semantic embeddings and sentiment scores for all lyrics."""
    initialize_text_models()
    print("\nProcessing Text Features...")
    df_processed = df.copy()
    lyrics_list = df_processed['lyrics'].fillna("").tolist()
    print(f"  > Generating semantic embeddings for {len(lyrics_list)} songs...")
    embeddings = embedding_model.encode(lyrics_list, show_progress_bar=True, batch_size=CONFIG["processing_batch_size"])
    df_processed['semantic_embedding'] = [emb.tolist() for emb in embeddings]
    print(f"  > Analyzing sentiment for {len(lyrics_list)} songs...")
    sentiment_results = sentiment_pipeline(lyrics_list, batch_size=CONFIG["processing_batch_size"], truncation=True)
    sentiment_vectors = []
    for result_list in sentiment_results:
        scores = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        result = result_list[0] if isinstance(result_list, list) else result_list
        label = result['label'].lower()
        scores[label] = result['score']
        sentiment_vectors.append([scores['negative'], scores['neutral'], scores['positive']])
    df_processed['sentiment_vector'] = sentiment_vectors
    print("  > Text processing complete.")
    return df_processed

def main():
    """Main function to run the entire preprocessing pipeline."""
    all_songs = load_and_consolidate_data(CONFIG["source_dir"])
    if not all_songs: return
    df = pd.DataFrame(all_songs)

    df, scalar_scaler, chord_map = process_metadata_features(df)
    df = process_text_features(df)

    mfcc_scaler = None
    if PROCESS_AUDIO_FILES:
        if 'audio_file_path' not in df.columns:
            print("\n[WARNING] PROCESS_AUDIO_FILES is True, but 'audio_file_path' column not found.")
            df['mfcc_vector'] = [[0.0] * CONFIG["n_mfcc"]] * len(df)
        else:
            df, mfcc_scaler = process_audio_files(df)
    else:
        print("\nSkipping raw audio file processing (MFCCs).")
        df['mfcc_vector'] = [[0.0] * CONFIG["n_mfcc"]] * len(df)

    print("\nAssembling final feature vectors...")
    final_data_to_save = []
    for _, row in df.iterrows():
        scalar_vec = row[CONFIG["scalar_audio_features"]].tolist()
        audio_features_vector = scalar_vec + row['harmonic_vector'] + row['mfcc_vector']
        text_features_vector = row['semantic_embedding'] + row['sentiment_vector']
        
        final_data_to_save.append({
            "title": row['title'],
            "artist": row['artist'],
            "main_genre": row['main_genre'],
            "popularity": row.get('popularity', 0),
            "chord_progression": row['chord_progression'],
            "lyrics": row.get('lyrics', ''),
            "audio_features_vector": audio_features_vector,
            "text_features_vector": text_features_vector
        })

    if final_data_to_save:
        audio_dim = len(final_data_to_save[0]['audio_features_vector'])
        text_dim = len(final_data_to_save[0]['text_features_vector'])
        print("\n--- Final Vector Dimensions ---")
        print(f"Audio Features Vector Dimension: {audio_dim}")
        print(f"Text Features Vector Dimension:  {text_dim}")
        print("---------------------------------")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    joblib.dump(scalar_scaler, os.path.join(CONFIG["output_dir"], CONFIG["scalar_scaler_filename"]))
    with open(os.path.join(CONFIG["output_dir"], CONFIG["chord_map_filename"]), 'w', encoding='utf-8') as f:
        json.dump(chord_map, f, indent=2)
    if mfcc_scaler:
        joblib.dump(mfcc_scaler, os.path.join(CONFIG["output_dir"], CONFIG["mfcc_scaler_filename"]))
    with open(os.path.join(CONFIG["output_dir"], CONFIG["output_filename"]), 'w', encoding='utf-8') as f:
        json.dump(final_data_to_save, f)
    print(f"\n✅ Preprocessing complete! Model-ready data saved to '{os.path.join(CONFIG['output_dir'], CONFIG['output_filename'])}'")

if __name__ == "__main__":
    main()