
import json
import os
import sys
import re
import librosa
import numpy as np
import logging
import warnings
import traceback
from pathlib import Path

# Optional: use pydub for MP3 loading
try:
    from pydub import AudioSegment
    logging.info("pydub imported successfully.")
except ImportError:
    logging.warning("pydub not found. Install with 'pip install pydub'. MP3 loading might fall back to librosa/ffmpeg.")
    AudioSegment = None

# scikit-image for resizing
try:
    from skimage.transform import resize
    logging.info("scikit-image imported successfully.")
except ImportError:
    logging.error("scikit-image not found. Install with 'pip install scikit-image'.")
    resize = None

# Pillow for image saving
try:
    from PIL import Image
    logging.info("Pillow imported successfully.")
except ImportError:
    logging.error("Pillow not found. Install with 'pip install Pillow'.")
    Image = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings('ignore', category=FutureWarning)

# Constants
TARGET_HEIGHT = 128
TARGET_WIDTH = 512
IMAGE_OUTPUT_SUBDIR = 'composite_images'

# --- Helper Functions ---

def load_song_data_from_json(json_path: Path) -> list:
    """Load a list of song metadata dicts from a JSON file."""
    try:
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                logging.info(f"Loaded {len(data)} entries from {json_path}")
                return data
            else:
                logging.error(f"JSON root is not a list in {json_path}")
                return []
    except Exception as e:
        logging.error(f"Failed to load JSON {json_path}: {e}")
        return []


def sanitize_filename(text: str) -> str:
    """Sanitize strings for filenames by replacing invalid chars."""
    if not text:
        return 'unknown'
    # Replace forbidden characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    sanitized = re.sub(r'\.$', '', sanitized)
    return sanitized or 'unknown'


def normalize_feature_array(arr: np.ndarray) -> np.ndarray:
    """Normalize a NumPy array to [0,1]."""
    min_val, max_val = np.min(arr), np.max(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def generate_composite_image(
    audio_path: Path,
    output_dir: Path,
    song_info: dict,
    target_height: int,
    target_width: int
) -> str | None:
    """
    Process an audio file to extract mel, mfcc, and chroma features,
    combine into an RGB image, and save to disk.
    Returns the saved path (relative) or None on failure.
    """
    try:
        if not audio_path.exists():
            logging.warning(f"Audio file not found: '{audio_path}'")
            return None

        # Load audio (librosa uses ffmpeg under the hood if available)
        y, sr = librosa.load(str(audio_path), sr=None)
        if y is None or y.size == 0:
            logging.warning(f"Empty audio signal for {audio_path}")
            return None

        # Optional silence trimming
        try:
            y, _ = librosa.effects.trim(y, top_db=20)
        except Exception:
            logging.debug(f"Silence trimming failed; using full signal for {audio_path}")

        # Feature extraction
        # 1. Mel spectrogram
        M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_height, fmax=8000)
        M_db = librosa.power_to_db(M, ref=np.max)

        # 2. MFCC
        n_mfcc = 20
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        except Exception:
            logging.warning(f"MFCC extraction failed for {audio_path}; using zeros")
            mfcc = np.zeros((n_mfcc, M_db.shape[1]))

        # 3. Chromagram
        n_chroma = 12
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        except Exception:
            logging.warning(f"Chroma extraction failed for {audio_path}; using zeros")
            chroma = np.zeros((n_chroma, M_db.shape[1]))

        # Resize features to target dimensions
        if resize is None:
            logging.error("Resizing unavailable—install scikit-image.")
            return None

        mel_resized = resize(M_db, (target_height, target_width), anti_aliasing=True)
        mfcc_resized = resize(mfcc, (target_height, target_width), anti_aliasing=True)
        chroma_resized = resize(chroma, (target_height, target_width), anti_aliasing=True)

        # Normalize each channel
        mel_norm = normalize_feature_array(mel_resized)
        mfcc_norm = normalize_feature_array(mfcc_resized)
        chroma_norm = normalize_feature_array(chroma_resized)

        # Stack into RGB image
        composite = np.stack([mel_norm, mfcc_norm, chroma_norm], axis=-1)
        composite_uint8 = (composite * 255).astype(np.uint8)

        if Image is None:
            logging.error("Pillow unavailable—cannot save images.")
            return None

        img = Image.fromarray(composite_uint8)

        # Build filename
        song_id = song_info.get('id', '')
        name = sanitize_filename(song_info.get('spotify_song_name', ''))
        artist = sanitize_filename(song_info.get('spotify_artist_name', ''))
        filename = f"{song_id}_{artist}_{name}.png"

        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / filename
        img.save(save_path)

        rel_path = os.path.relpath(save_path, Path.cwd())
        logging.info(f"Saved composite image: {rel_path}")
        return rel_path

    except Exception:
        logging.error(f"Error processing {audio_path}: {traceback.format_exc()}")
        return None


# --- Main Execution ---
if __name__ == '__main__':
    # Reconstruct JSON path from all CLI args (handles spaces without quotes)


    raw_path = ' '.join("/home/psyph3ri0n/Documents/projects-2025/sur/downloads/qw/song_data.json").strip()
    json_path = Path(raw_path).expanduser().resolve()

    if not json_path.is_file():
        print(f"Error: JSON file not found at '{json_path}'")
        sys.exit(1)

    song_list = load_song_data_from_json(json_path)
    if not song_list:
        print("No song entries found. Exiting.")
        sys.exit(1)

    # Prepare output directory
    output_dir = json_path.parent / IMAGE_OUTPUT_SUBDIR

    # Process each song
    total = len(song_list)
    success = 0
    fail = 0
    for idx, info in enumerate(song_list, start=1):
        fp = str(Path(info.get('file_path', '')).expanduser().resolve())
        result = generate_composite_image(
            Path(fp), output_dir, info, TARGET_HEIGHT, TARGET_WIDTH
        )
        if result:
            success += 1
        else:
            fail += 1
        logging.info(f"[{idx}/{total}] Success: {success}, Fail: {fail}")

    print(f"Finished: {success}/{total} images generated. Failed: {fail}.")
    print(f"Images saved in: {output_dir}")
