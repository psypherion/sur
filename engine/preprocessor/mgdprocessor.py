import os
import csv
import json
import logging
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

class _MGDLoader:
    """Loads configuration and environment variables for data processing."""
    def __init__(self) -> None:
        self.MGD_DB = os.getenv("MGD_DB")
        if not self.MGD_DB:
            raise ValueError("MGD_DB path not found in environment variables.")
        self.MGD_YEARS = os.getenv("MGD_YEARS")
        if not self.MGD_YEARS:
            raise ValueError("MGD_YEARS not found in environment variables.")
        self.MGD_COUNTRIES = os.getenv("MGD_COUNTRIES")
        if not self.MGD_COUNTRIES:
            raise ValueError("MGD_COUNTRIES not found in environment variables.")

class _CompareHITSandAll:
    """Compares two song lists and finds common entries."""
    def __init__(self, all_songs: List[Dict], hit_songs: List[Dict]) -> None:
        self.all_songs = all_songs
        self.hit_songs = hit_songs

    def compare(self) -> List[Dict]:
        """Return list of songs present in both all_songs and hit_songs, matched by artist and title."""
        hit_set = { (song['artist'], song['title']) for song in self.hit_songs }
        common_songs = [song for song in self.all_songs if (song['artist'], song['title']) in hit_set]
        logging.info(f"Total common songs found: {len(common_songs)}")
        logging.info(f"Total all songs: {len(self.all_songs)}")
        logging.info(f"Total hit songs: {len(self.hit_songs)}")
        return common_songs

class MGDProcessor:
    """Processes general chart data, across years and chart countries."""
    def __init__(self) -> None:
        loader = _MGDLoader()
        self.db_path: str = loader.MGD_DB or ""
        self.years: List[str] = json.loads(loader.MGD_YEARS or "[]")
        self.countries: List[str] = json.loads(loader.MGD_COUNTRIES or "[]")

    def chart_paths(self, year: str, country: str) -> List[str]:
        """Return a list of file paths for all weekly country charts for a given year and country."""
        dir_path = os.path.join(self.db_path, country, year)
        full_dir = os.path.join(os.getcwd(), dir_path)
        if not os.path.isdir(full_dir):
            logging.warning(f"Directory not found: {full_dir}")
            return []
        weeklycharts = [os.path.join(full_dir, f) for f in os.listdir(full_dir) if os.path.isfile(os.path.join(full_dir, f))]
        return weeklycharts
    
    def load_charts(self, year: str, country: str) -> List[Dict[str, str]]:
        """Load all weekly chart files for a specific year and chart country, returning a list of song dicts."""
        paths = self.chart_paths(year, country)
        all_songs: List[Dict[str, str]] = []
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        song_info = {
                            "artist": row.get("artist", "").strip(),
                            "title": row.get("song_name", "").strip(),
                            "spotify_id": row.get("song_id", "").strip()
                        }
                        all_songs.append(song_info)
            except FileNotFoundError:
                logging.error(f"File not found: {path}")
        unique_songs = { (song['artist'], song['title'], song['spotify_id']): song for song in all_songs }
        return list(unique_songs.values())
    
    def load_all(self) -> List[Dict[str, str]]:
        """Load all unique songs from all years and countries."""
        all_songs: List[Dict[str, str]] = []
        for year in self.years:
            logging.info(f"Loading songs for year: {year}")
            for country in self.countries:
                logging.info(f"  Chart country: {country}")
                songs = self.load_charts(year, country)
                all_songs.extend(songs)
        unique_songs = { (song['artist'], song['title'], song['spotify_id']): song for song in all_songs }
        return list(unique_songs.values())

class MGDHitsProcessor(MGDProcessor):
    """Processes the hit songs data."""
    def __init__(self) -> None:
        super().__init__()
        hits_env: str = os.getenv("MGD_HITS", "")
        if not hits_env:
            raise ValueError("MGD_HITS path not found in environment variables.")
        if not os.path.isabs(hits_env):
            root_dir = os.path.join(os.getcwd(), self.db_path.split("Charts")[0])
            self.MGD_HITS = os.path.normpath(os.path.join(root_dir, hits_env))
        else:
            self.MGD_HITS = hits_env

    def load_hits(self) -> List[Dict[str, str]]:
        """Load all hit songs from the hits TSV file."""
        all_songs: List[Dict[str, str]] = []
        try:
            with open(self.MGD_HITS, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    song_info = {
                        "artist": row.get("artist_name", "").strip(),
                        "title": row.get("song_name", "").strip(),
                        "spotify_id": row.get("song_id", "").strip(),
                        "acousticness": row.get("acousticness", ""),
                        "danceability": row.get("danceability", ""),
                        "energy": row.get("energy", ""),
                        "instrumentalness": row.get("instrumentalness", ""),
                        "liveness": row.get("liveness", ""),
                        "loudness": row.get("loudness", ""),
                        "speechiness": row.get("speechiness", ""),
                        "tempo": row.get("tempo", ""),
                        "valence": row.get("valence", ""),
                        "duration_ms": row.get("duration_ms", ""),
                        "time_signature": row.get("time_signature", ""),
                        "key": row.get("key", ""),
                        "mode": row.get("mode", "")
                    }
                    all_songs.append(song_info)
        except FileNotFoundError:
            logging.error(f"Hit songs file not found: {self.MGD_HITS}")
        return all_songs

__all__ = ["MGDProcessor", "MGDHitsProcessor"]

if __name__ == "__main__":
    processor = MGDProcessor()
    hit_processor = MGDHitsProcessor()
    all_songs = processor.load_all()
    logging.info(f"Total unique songs loaded from all countries: {len(all_songs)}")
    hit_songs = hit_processor.load_hits()
    logging.info(f"Total unique hit songs loaded: {len(hit_songs)}")
    logging.info(f"Top 5 songs from hit songs: {hit_songs[:5]}")