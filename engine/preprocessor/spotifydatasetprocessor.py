import os
import csv
import logging
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

class _SpotifyDatasetLoader:
    """Loads configuration and environment variables for Spotify dataset processing."""
    def __init__(self) -> None:
        self.SPOTIFY_CSV = os.getenv("SPOTIFY_CSV")
        if not self.SPOTIFY_CSV:
            raise ValueError("SPOTIFY_CSV path not found in environment variables.")
        self.spotify_csv = os.path.join(os.getcwd(), self.SPOTIFY_CSV)
        if not os.path.isfile(self.spotify_csv):
            raise FileNotFoundError(f"Spotify CSV file not found at path: {self.spotify_csv}")
        logging.info(f"Spotify CSV path set to: {self.spotify_csv}")
    
class SpotifyDatasetProcessor:
    """Processes the Spotify dataset CSV file."""
    def __init__(self) -> None:
        loader = _SpotifyDatasetLoader()
        self.spotify_csv: str = loader.spotify_csv

    def load_spotify_data(self) -> List[Dict[str, str]]:
        """Load and return the Spotify dataset as a list of dictionaries."""
        spotify_data = []
        try:
            with open(self.spotify_csv, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    spotify_data.append({
                        "spotify_id": row.get("id", "").strip(),
                        "title": row.get("name", "").strip(),
                        "artist": row.get("artists", "").strip(),
                        "danceability": row.get("danceability", "").strip(),
                        "energy": row.get("energy", "").strip(),
                        "key": row.get("key", "").strip(),
                        "loudness": row.get("loudness", "").strip(),
                        "mode": row.get("mode", "").strip(),
                        "speechiness": row.get("speechiness", "").strip(),
                        "acousticness": row.get("acousticness", "").strip(),
                        "instrumentalness": row.get("instrumentalness", "").strip(),
                        "liveness": row.get("liveness", "").strip(),
                        "valence": row.get("valence", "").strip(),
                        "tempo": row.get("tempo", "").strip(),
                    })
            logging.info(f"Total records loaded from Spotify dataset: {len(spotify_data)}")
            return spotify_data
        except FileNotFoundError:
            logging.error(f"Spotify dataset file not found: {self.spotify_csv}")
            return []
        
__all__ = ["SpotifyDatasetProcessor"]

if __name__ == "__main__":
    processor = SpotifyDatasetProcessor()
    data = processor.load_spotify_data()
    logging.info(f"Sample data: {data[:5]}")
