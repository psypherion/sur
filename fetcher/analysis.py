import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import warnings
# Try importing pydub; it's often needed by librosa for MP3s,
# or you can use it to convert MP3 to WAV before librosa.load
try:
    from pydub import AudioSegment
except ImportError:
    logging.warning("pydub not found. Install with 'pip install pydub'. MP3 loading might fail without it or ffmpeg.")
    AudioSegment = None # Define AudioSegment as None if import fails

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore', category=FutureWarning) # Suppress some future warnings from libraries

# Initialize NLTK VADER sentiment analyzer
try:
    analyzer = SentimentIntensityAnalyzer()
    logging.info("NLTK VADER SentimentIntensityAnalyzer initialized.")
except LookupError:
    logging.warning("NLTK Vader lexicon not found. Downloading...")
    try:
        nltk.download('vader_lexicon')
        analyzer = SentimentIntensityAnalyzer() # Try initializing again after download
        logging.info("NLTK Vader lexicon downloaded and analyzer initialized.")
    except Exception as e:
        logging.error(f"Failed to download NLTK Vader lexicon: {e}. Sentiment analysis will be skipped.", exc_info=True)
        analyzer = None


# --- Helper Functions (from previous steps) ---

def load_song_data_from_json(json_filepath: str) -> list:
    """Loads song data (full list of dictionaries) from a JSON file."""
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                 logging.info(f"Successfully loaded {len(data)} entries from '{json_filepath}'.")
                 return data
            else:
                 logging.error(f"JSON data in '{json_filepath}' is not a list.")
                 return [] # Return empty list if data is not a list

    except FileNotFoundError:
        logging.error(f"Error: JSON file not found at '{json_filepath}'")
        return [] # Return empty list on file not found
    except json.JSONDecodeError:
        logging.error(f"Error: Could not parse JSON file at '{json_filepath}'")
        return [] # Return empty list on JSON parsing error
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading '{json_filepath}': {e}", exc_info=True)
        return [] # Return empty list on other errors

def flatten_songdata_info(data_list):
    """Flattens the nested songdata_info into top-level keys."""
    flattened_data = []
    for item in data_list:
        new_item = item.copy()
        songdata_info = new_item.pop('songdata_info', {})

        for key, value in songdata_info.items():
             # Prefix songdata keys to avoid collision, except for track/artist which are handled separately
             # Actually, the merge function already prefixes them like 'songdata_track'. Let's keep that.
             new_item[f'songdata_{key}'] = value # Ensure keys are prefixed

        flattened_data.append(new_item)
    return flattened_data

def analyze_audio_file(audio_filepath):
    """Loads an audio file and extracts features using librosa."""
    # Ensure the file path is absolute or relative to the script's working directory
    absolute_audio_filepath = os.path.abspath(audio_filepath)

    try:
        # Librosa can often load MP3s directly if FFmpeg is installed and discoverable
        # If this fails, you might need to use pydub to convert to wav first
        y, sr = librosa.load(absolute_audio_filepath, sr=None) # Use original sample rate


        # --- Feature Extraction ---
        duration = librosa.get_duration(y=y, sr=sr)

        # Handle potential issues with beat tracking on very short or silent files
        tempo = np.nan # Default to NaN if tempo detection fails
        try:
             # Only attempt beat tracking if duration is reasonable
             if duration > 2: # Arbitrary threshold, adjust as needed
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        except Exception:
             logging.debug(f"Tempo detection failed for {audio_filepath}.")


        # Calculate other features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)) if y.size > 0 else np.nan
        energy = np.mean(librosa.feature.rmse(y=y)) if y.size > 0 else np.nan
        # Compute MFCCs and take mean (handle case where y is too short)
        mfccs_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1).tolist() if y.size > (2048 + 512) * 2 else [np.nan] * 20 # Requires minimum audio length


        audio_features = {
            'audio_duration_sec': duration,
            'audio_sample_rate': sr,
            'audio_tempo_bpm': tempo,
            'audio_spectral_centroid': spectral_centroid,
            'audio_energy_rmse': energy,
            'audio_mfcc_mean': mfccs_mean # Store as list
        }
        # Optionally return y, sr for potential visualization later if needed
        # return audio_features, y, sr
        return audio_features

    except FileNotFoundError:
        logging.warning(f"Audio file not found for analysis: {absolute_audio_filepath}")
        return None
    except Exception as e:
        logging.error(f"Error analyzing audio file {absolute_audio_filepath}: {e}", exc_info=True)
        return None

def analyze_lyrics_file(lyrics_filepath):
    """Loads a lyrics file and extracts features using NLTK."""
    # Ensure the file path is absolute or relative to the script's working directory
    absolute_lyrics_filepath = os.path.abspath(lyrics_filepath)

    try:
        with open(absolute_lyrics_filepath, 'r', encoding='utf-8') as f:
            lyrics_text = f.read()

        # --- Basic Text Processing ---
        words = nltk.word_tokenize(lyrics_text)
        sentences = nltk.sent_tokenize(lyrics_text)

        # Remove punctuation and convert to lowercase for word counts
        words_cleaned = [word.lower() for word in words if word.isalpha()]

        # --- Feature Extraction ---
        word_count = len(words_cleaned)
        unique_word_count = len(set(words_cleaned))
        sentence_count = len(sentences)
        type_token_ratio = (unique_word_count / word_count) if word_count > 0 else 0

        # Sentiment Analysis (using VADER)
        sentiment_scores = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0} # Default if analyzer is not available or text is empty
        if analyzer and lyrics_text.strip():
             sentiment_scores = analyzer.polarity_scores(lyrics_text)


        lyrics_features = {
            'lyrics_word_count': word_count,
            'lyrics_unique_word_count': unique_word_count,
            'lyrics_sentence_count': sentence_count,
            'lyrics_type_token_ratio': type_token_ratio,
            'lyrics_sentiment_neg': sentiment_scores['neg'],
            'lyrics_sentiment_neu': sentiment_scores['neu'],
            'lyrics_sentiment_pos': sentiment_scores['pos'],
            'lyrics_sentiment_compound': sentiment_scores['compound']
        }
        return lyrics_features

    except FileNotFoundError:
        logging.warning(f"Lyrics file not found for analysis: {absolute_lyrics_filepath}")
        return None
    except Exception as e:
        logging.error(f"Error analyzing lyrics file {absolute_lyrics_filepath}: {e}", exc_info=True)
        return None


# --- Main Analysis Function ---
def perform_analysis(json_filepath: str):
    """
    Performs analysis and visualization on the dataset from the JSON file,
    including analyzing linked audio and lyrics files.
    """
    logging.info("--- Starting Data Analysis ---")
    logging.info(f"Loading data from: {json_filepath}")

    # 1. Load Data
    all_song_data = load_song_data_from_json(json_filepath)
    if not all_song_data:
        logging.error("No song data loaded from JSON or JSON is empty/invalid. Analysis aborted.")
        return # Exit if loading fails

    # 2. Flatten Songdata Info and Create DataFrame
    flattened_song_data = flatten_songdata_info(all_song_data)
    df = pd.DataFrame(flattened_song_data)

    # Convert songdata numerical columns to numeric types
    # Columns are prefixed like 'songdata_bpm' after flattening
    songdata_numeric_cols_base = ['bpm', 'duration', 'acousticness', 'danceability',
                                  'energy', 'instrumentalness', 'liveness', 'loudness',
                                  'speechiness', 'valence', 'popularity']
    for col_base in songdata_numeric_cols_base:
        col = f'songdata_{col_base}'
        if col in df.columns:
             # Use errors='coerce' to turn unparseable values into NaN
             df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure 'id' is numeric if it exists
    if 'id' in df.columns:
        df['id'] = pd.to_numeric(df['id'], errors='coerce')


    # 3. Audio Analysis & Add Features to DataFrame
    logging.info("\nStarting audio file analysis...")
    audio_features_list = [] # To collect features if adding separately is preferred

    # Add columns for audio features initialized to NaN
    audio_feature_cols = ['audio_duration_sec', 'audio_sample_rate', 'audio_tempo_bpm',
                          'audio_spectral_centroid', 'audio_energy_rmse', 'audio_mfcc_mean']
    for col in audio_feature_cols:
         if col not in df.columns: # Avoid adding if they somehow exist
             df[col] = np.nan # Initialize with NaN


    for index, row in df.iterrows():
        audio_file_path_relative = row.get('file_path') # This path is relative from the download script

        if audio_file_path_relative:
             # Construct absolute path based on script's current working directory
             audio_file_path = os.path.abspath(audio_file_path_relative)

             if os.path.exists(audio_file_path):
                logging.info(f"Analyzing audio for {row.get('spotify_artist_name', 'Unknown')} - {row.get('spotify_song_name', 'Unknown')}")
                features = analyze_audio_file(audio_file_path) # analyze_audio_file only returns features dict now

                if features:
                     # Add features directly to the DataFrame row
                     for key, value in features.items():
                          df.loc[index, key] = value # Use the key from the features dict (already prefixed)

                else:
                     logging.warning(f"Failed to extract audio features for {row.get('spotify_song_name', 'Unknown')}. Corresponding columns for this row will be NaN.")

             else:
                 logging.warning(f"Audio file not found at expected path for analysis: {audio_file_path}")
                 # Corresponding audio feature columns for this row will remain NaN

        else:
             logging.warning(f"Audio file path missing in JSON for index {index}. Skipping audio analysis for this entry.")
             # Corresponding audio feature columns for this row will remain NaN


    # 4. Lyrics Analysis & Add Features to DataFrame
    logging.info("\nStarting lyrics analysis...")
    lyrics_feature_cols = ['lyrics_word_count', 'lyrics_unique_word_count', 'lyrics_sentence_count',
                           'lyrics_type_token_ratio', 'lyrics_sentiment_neg', 'lyrics_sentiment_neu',
                           'lyrics_sentiment_pos', 'lyrics_sentiment_compound']
    for col in lyrics_feature_cols:
         if col not in df.columns: # Avoid adding if they somehow exist
             df[col] = np.nan # Initialize with NaN


    for index, row in df.iterrows():
        lyrics_file_path_relative = row.get('lyrics_file_path') # This path is relative from the lyrics script

        if lyrics_file_path_relative:
            # Construct absolute path based on script's current working directory
            lyrics_file_path = os.path.abspath(lyrics_file_path_relative)

            if os.path.exists(lyrics_file_path):
                logging.info(f"Analyzing lyrics for {row.get('spotify_artist_name', 'Unknown')} - {row.get('spotify_song_name', 'Unknown')}")
                features = analyze_lyrics_file(lyrics_file_path) # analyze_lyrics_file only returns features dict now

                if features:
                     # Add features directly to the DataFrame row
                     for key, value in features.items():
                          df.loc[index, key] = value # Use the key from the features dict (already prefixed)
                else:
                     logging.warning(f"Failed to extract lyrics features for {row.get('spotify_song_name', 'Unknown')}. Corresponding columns for this row will be NaN.")
            else:
                logging.warning(f"Lyrics file not found at expected path for analysis: {lyrics_file_path}")
                # Corresponding lyrics feature columns for this row will remain NaN

        else:
             logging.warning(f"Lyrics file path missing in JSON for index {index}. Skipping lyrics analysis for this entry.")
             # Corresponding lyrics feature columns for this row will remain NaN


    # Convert added audio/lyrics numeric columns to numeric types
    # These columns are now added dynamically, so fetch their names
    all_numeric_features = [col for col in df.columns if col.startswith(('songdata_', 'audio_', 'lyrics_')) and col not in ['audio_mfcc_mean', 'audio_sample_rate']]
    if 'id' in df.columns: all_numeric_features.append('id') # Include id if needed

    for col in all_numeric_features:
        if col in df.columns: # Double check column exists
            df[col] = pd.to_numeric(df[col], errors='coerce')


    # --- Perform Analysis and Visualizations using the final DataFrame 'df' ---
    logging.info("\nPerforming analysis and visualizations...")

    print("\nDataFrame Info after Analysis:")
    df.info()
    print("\nDataFrame Head after Analysis:")
    print(df.head())

    # Display descriptive statistics for all numeric columns
    print("\nDescriptive Statistics for All Numeric Features:")
    print(df[all_numeric_features].describe())


    # --- Visualizations ---
    # Check if DataFrame has enough data to plot
    if df.shape[0] < 2 or df[all_numeric_features].dropna().shape[0] < 2:
         logging.warning("Not enough complete data points for robust visualization after analysis. Skipping plots.")
    else:
        try:
            sns.set_style("whitegrid")

            # Example 1: Metadata Visualizations (Histograms, Scatter)
            fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))
            fig1.suptitle("Metadata Analysis", fontsize=16)

            if 'songdata_bpm' in df.columns and df['songdata_bpm'].dropna().size > 0:
                sns.histplot(df['songdata_bpm'].dropna(), kde=True, ax=axes1[0, 0])
                axes1[0, 0].set_title("Distribution of BPM (Songdata)")

            if 'songdata_energy' in df.columns and 'songdata_danceability' in df.columns and not df[['songdata_energy', 'songdata_danceability']].dropna().empty:
                sns.scatterplot(data=df.dropna(subset=['songdata_energy', 'songdata_danceability']), x='songdata_energy', y='songdata_danceability', alpha=0.6, ax=axes1[0, 1])
                axes1[0, 1].set_title("Energy vs Danceability (Songdata)")

            if 'songdata_key' in df.columns and df['songdata_key'].dropna().size > 0:
                # Get top N keys, handle cases with fewer than N unique keys
                top_keys = df['songdata_key'].value_counts().nlargest(min(5, df['songdata_key'].nunique()))
                if not top_keys.empty:
                    sns.barplot(x=top_keys.index, y=top_keys.values, palette="viridis", ax=axes1[1, 0])
                    axes1[1, 0].set_title("Top Keys (Songdata)")
                    axes1[1, 0].set_xlabel("Key")
                    axes1[1, 0].set_ylabel("Count")
                else:
                     axes1[1, 0].set_title("Top Keys (Songdata)") # Title even if no data
                     axes1[1, 0].text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=axes1[1, 0].transAxes)


            if 'songdata_popularity' in df.columns and df['songdata_popularity'].dropna().size > 0:
                sns.histplot(df['songdata_popularity'].dropna(), kde=True, bins=max(5, int(df['songdata_popularity'].dropna().nunique()/2)), ax=axes1[1, 1]) # Adjust bins
                axes1[1, 1].set_title("Distribution of Popularity (Songdata)")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

            # Example 2: Combined Feature Visualizations
            # Only plot if relevant columns exist and have data
            combined_plot_cols = [('songdata_bpm', 'audio_tempo_bpm'), ('songdata_valence', 'lyrics_sentiment_compound')]
            valid_combined_plots = [cols for cols in combined_plot_cols if cols[0] in df.columns and cols[1] in df.columns and not df[list(cols)].dropna().empty]

            if valid_combined_plots:
                 fig2, axes2 = plt.subplots(1, len(valid_combined_plots), figsize=(7 * len(valid_combined_plots), 6))
                 if len(valid_combined_plots) == 1: # Ensure axes2 is iterable
                      axes2 = [axes2]

                 fig2.suptitle("Combined Feature Analysis", fontsize=16)

                 for i, (x_col, y_col) in enumerate(valid_combined_plots):
                      sns.scatterplot(data=df.dropna(subset=[x_col, y_col]), x=x_col, y=y_col, alpha=0.6, ax=axes2[i])
                      axes2[i].set_title(f"{x_col.replace('songdata_', '').replace('audio_', '').replace('lyrics_', '')} vs {y_col.replace('songdata_', '').replace('audio_', '').replace('lyrics_', '')}")
                      axes2[i].set_xlabel(x_col)
                      axes2[i].set_ylabel(y_col)

                 plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                 plt.show()
            else:
                 logging.warning("Not enough data for combined feature scatter plots.")


            # Example 3: Correlation Heatmap for All Numeric Features
            # Filter for columns that were successfully converted to numeric
            plottable_numeric_cols = df[all_numeric_features].dropna(axis=1, how='all').columns.tolist() # Drop columns that are all NaN

            if plottable_numeric_cols and len(plottable_numeric_cols) > 1: # Need at least 2 columns for a heatmap
                plt.figure(figsize=(max(10, len(plottable_numeric_cols)), max(8, len(plottable_numeric_cols)*0.8))) # Adjust figure size based on number of columns
                correlation_matrix_all = df[plottable_numeric_cols].corr()
                sns.heatmap(correlation_matrix_all, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
                plt.title("Feature Correlation Heatmap (Metadata, Audio, Lyrics Features)")
                plt.tight_layout()
                plt.show()
            else:
                 logging.warning("Not enough numeric columns with data for correlation heatmap.")

        except Exception as e:
             logging.error(f"An error occurred during visualization: {e}", exc_info=True)
             print(f"An error occurred during visualization: {e}")


    logging.info("--- Data Analysis Complete ---")


if __name__ == "__main__":
    print("--- Starting Data Analysis Script ---")
    print("This script analyzes data from a song_data.json file and linked audio/lyrics files.")
    print("Prerequisites:")
    print("1. Ensure you have run the previous data collection scripts to generate song_data.json, .mp3, and .txt files.")
    print("2. Ensure required Python libraries (pandas, numpy, matplotlib, seaborn, librosa, soundfile, pydub, nltk, scipy) are installed.")
    print("3. Ensure FFmpeg is installed and accessible in your system's PATH for audio analysis.")
    print("4. Ensure NLTK data (like 'vader_lexicon') is downloaded (run python and `nltk.download('vader_lexicon')`).")
    print("Usage: python your_analysis_script_name.py <path_to_song_data.json>")



    input_json_path = "/home/psyph3ri0n/Documents/projects-2025/sur/downloads/My top tracks playlist/song_data.json"

    # Check if the provided path is a file and exists
    if not os.path.isfile(input_json_path):
        print(f"\nError: The provided path '{input_json_path}' is not a valid file or does not exist.")
        sys.exit(1)

    perform_analysis(input_json_path)

    print("\n--- Script Execution Finished ---")