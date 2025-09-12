import os
import csv
import json
from dotenv import load_dotenv
from typing import List, Dict
load_dotenv()

class _MGDLoader:
    def __init__(self) -> None:
        self.MGD_DB = os.getenv("MGD_DB")
        if not self.MGD_DB:
            raise ValueError("MGD_DB path not found in environment variables.")
        self.MGD_YEARS = os.getenv("MGD_YEARS")
        if not self.MGD_YEARS:
            raise ValueError("MGD_YEARS not found in environment variables.")
        self.MGD_TYPES = os.getenv("MGD_TYPES")
        if not self.MGD_TYPES:
            raise ValueError("MGD_TYPES not found in environment variables.")

class _CompareHITSandAll:
    def __init__(self, all_songs: List[Dict], hit_songs: List[Dict]) -> None:
        self.all_songs = all_songs
        self.hit_songs = hit_songs

    def compare(self) -> List[Dict]:
        hit_set = { (song['artist'], song['title']): song for song in self.hit_songs }
        common_songs = [song for song in self.all_songs if (song['artist'], song['title']) in hit_set]
        print(f"Total common songs found: {len(common_songs)}")
        print(f"Total all songs: {len(self.all_songs)}")
        print(f"Total hit songs: {len(self.hit_songs)}")
        return common_songs

class MGDProcessor:
    def __init__(self) -> None:
        loader = _MGDLoader()
        self.db_path: str = loader.MGD_DB or ""
        self.years = json.loads(loader.MGD_YEARS or "[]")
        self.types = json.loads(loader.MGD_TYPES or "[]")

    def chart_paths(self, year: str, chart_type: str) -> List[str]:
        weeklycharts = [os.path.join(os.getcwd(), self.db_path, chart_type, year, f) 
                        for f in os.listdir(os.path.join(self.db_path, chart_type, year))]
        return weeklycharts
    
    def load_charts(self, year: str, chart_type: str) -> List[Dict[str, str]]:
        paths = self.chart_paths(year, chart_type)
        all_songs: List[Dict] = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    song_info = {
                        "artist": row.get("artist", "").strip(),
                        "title": row.get("song_name", "").strip(),
                        "spotify_id": row.get("song_id", "").strip()
                    }
                    all_songs.append(song_info)
        unique_songs = { (song['artist'], song['title'], song['spotify_id']): song for song in all_songs }
        songs = list(unique_songs.values())
        return songs
    
    def load_all(self) -> List[Dict[str, str]]:
        all_songs: List[Dict] = []
        for year in self.years:
            print(f"Loading songs for year: {year}")
            for chart_type in self.types:
                print(f"  Chart type: {chart_type}")
                songs = self.load_charts(year, chart_type)
                all_songs.extend(songs)
        unique_songs = { (song['artist'], song['title'], song['spotify_id']): song for song in all_songs }
        songs = list(unique_songs.values())
        return songs

class MGDHitsProcessor(MGDProcessor):
    def __init__(self) -> None:
        super().__init__()
        hits_env: str = os.getenv("MGD_HITS") or ""
        if not os.path.isabs(hits_env):
            self.MGD_HITS: str = os.path.join(os.getcwd(), self.db_path.split("Charts")[0], hits_env)
        else:
            self.MGD_HITS: str = hits_env
        if not self.MGD_HITS:
            raise ValueError("MGD_HITS path not found in environment variables.")

    def load_hits(self) -> List[Dict[str, str]]:
        all_songs: List[Dict] = []
        with open(self.MGD_HITS, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                song_info = {
                    "artist": row.get("artist_name", ""),
                    "title": row.get("song_name", "").strip(),
                    "spotify_id": row.get("song_id", "").strip(),
                    "acuosticness": row.get("acousticness", ""),
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
        return all_songs
    
__all__ = ["MGDProcessor", "MGDHitsProcessor"]

if __name__ == "__main__":
    processor = MGDProcessor()
    hit_processor = MGDHitsProcessor()
    all_songs = processor.load_all()
    print(f"Total unique songs loaded from all charts: {len(all_songs)}")
    hit_songs = hit_processor.load_hits()
    print(f"Total unique hit songs loaded: {len(hit_songs)}")
    print(f"Top 5 songs from hit songs: {hit_songs[:5]}")