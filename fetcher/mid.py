import json
import os

# --- Paths ---
PLAYLIST_JSON = "playlist_data.json"
SONG_DETAILS_JSON = "downloads/spoti-test/song_details.json"

# --- Load Data ---
with open(PLAYLIST_JSON, "r", encoding="utf-8") as f:
    playlist = json.load(f)

with open(SONG_DETAILS_JSON, "r", encoding="utf-8") as f:
    songs = json.load(f)

# --- Build lookup from playlist_data ---
# Key on (track.lower(), artist.lower())
playlist_map = {
    (entry["track"].lower().strip(), entry["artist"].lower().strip()): entry
    for entry in playlist
}

# --- Merge metadata into song_details entries ---
enriched = []
for song in songs:
    key = (song["spotify_song_name"].lower().strip(),
           song["spotify_artist_name"].lower().strip())
    meta = playlist_map.get(key)
    if not meta:
        print(f"⚠️ No metadata match for: {song['spotify_song_name']} by {song['spotify_artist_name']}")
        enriched.append(song)
        continue

    # Copy original fields, then extend with all metadata fields
    merged = dict(song)  # id, spotify_song_name, spotify_artist_name, youtube_title, file_path
    # Add or overwrite with metadata fields from playlist_data
    for field in [
        "key", "bpm", "duration", "camelot", "acousticness",
        "danceability", "energy", "instrumentalness", "liveness",
        "loudness", "speechiness", "valence", "popularity",
        "release_date", "track_url"
    ]:
        merged[field] = meta.get(field)
    enriched.append(merged)

# --- Write back ---
os.makedirs(os.path.dirname(SONG_DETAILS_JSON), exist_ok=True)
with open(SONG_DETAILS_JSON, "w", encoding="utf-8") as f:
    json.dump(enriched, f, indent=2, ensure_ascii=False)

print(f"✅ Enriched {len(enriched)} songs and wrote to {SONG_DETAILS_JSON}")
