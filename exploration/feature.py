#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# === CONFIGURATION ===
AUDIO_DIR    = "/home/psyph3ri0n/Documents/projects-2025/sur/downloads/qw/songs"
OUT_NUM_DIR  = "/home/psyph3ri0n/Documents/projects-2025/sur/features/numerical"
OUT_IMG_DIR  = "/home/psyph3ri0n/Documents/projects-2025/sur/features/mel_images"
SAVE_IMAGE   = True
SR           = 22050
HOP_LENGTH   = 512
WIN_LENGTH   = 2048
TOP_DB       = 30
N_MELS       = 128
N_MFCC       = 13
MAX_WORKERS  = 4

# ensure output directories exist
os.makedirs(OUT_NUM_DIR, exist_ok=True)
if SAVE_IMAGE:
    os.makedirs(OUT_IMG_DIR, exist_ok=True)

def extract_audio_features(audio_path: str) -> dict:
    """ Load, trim silence, and extract a suite of audio features. """
    # load & trim
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)

    # mel spectrogram (power -> dB)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # MFCCs from mel dB
    mfcc = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=N_MFCC)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)

    # Tonnetz on harmonic component
    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)

    # Temporal stats
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # aggregate means & stds
    def agg(feat):
        return feat.mean(axis=1), feat.std(axis=1)

    mfcc_mean, mfcc_std     = agg(mfcc)
    chroma_mean, chroma_std = agg(chroma)
    tonnetz_mean, tonnetz_std = agg(tonnetz)

    return {
        "mel": mel_db,
        "mfcc_mean": mfcc_mean, "mfcc_std": mfcc_std,
        "chroma_mean": chroma_mean, "chroma_std": chroma_std,
        "tonnetz_mean": tonnetz_mean, "tonnetz_std": tonnetz_std,
        "rms_mean": float(rms.mean()), "rms_std": float(rms.std()),
        "zcr_mean": float(zcr.mean()), "zcr_std": float(zcr.std()),
    }

def save_mel_image(mel_db: np.ndarray, img_path: str):
    """ Save a Mel spectrogram as a PNG. """
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(
        mel_db, sr=SR, hop_length=HOP_LENGTH, 
        x_axis="time", y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def process_file(fp: str) -> str:
    """ Extract features and save them for one file. """
    try:
        feats = extract_audio_features(fp)
        stem = Path(fp).stem.replace(" ", "_").replace("'", "")
        
        # save numerical features as .npz
        out_npz = os.path.join(OUT_NUM_DIR, f"{stem}.npz")
        np.savez(out_npz, **{k: v for k, v in feats.items() if k != "mel"})


        img_fp = os.path.join(OUT_IMG_DIR, f"{stem}.png")
        save_mel_image(feats["mel"], img_fp)

        return f"✅ {stem}"
    except Exception as e:
        return f"❌ {Path(fp).name}: {e}"

def main():
    # gather all MP3s
    files = [
        os.path.join(AUDIO_DIR, f)
        for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(".mp3")
    ]

    # parallel processing
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for status in tqdm(pool.map(process_file, files), total=len(files)):
            print(status)

if __name__ == "__main__":
    main()
