from mgdprocessor import MGDHitsProcessor
from spotifydatasetprocessor import SpotifyDatasetProcessor
from typing import List, Dict
import csv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

class DataPreprocessor:
    """Main class to handle data preprocessing from various datasets."""
    def __init__(self) -> None:
        self.mgd_processor = MGDHitsProcessor()
        self.spotify_processor = SpotifyDatasetProcessor()

    def load_all_data(self) -> List[Dict[str, str]]:
        """Load and combine data from MGD hits and Spotify dataset."""
        mgd_songs = self.mgd_processor.load_hits()
        spotify_songs = self.spotify_processor.load_spotify_data()

        combined_songs = {
            (song['artist'], song['title'], song['spotify_id'],
             song['danceability'], song['energy'], song['key'],
             song['loudness'], song['mode'], song['speechiness'],
             song['acousticness'], song['instrumentalness'],
             song['liveness'], song['valence'], song['tempo']): song for song in mgd_songs
        }

        for song in spotify_songs:
            key = (song['artist'], song['title'], song['spotify_id'],
                   song['danceability'], song['energy'], song['key'],
                   song['loudness'], song['mode'], song['speechiness'],
                   song['acousticness'], song['instrumentalness'],
                   song['liveness'], song['valence'], song['tempo'])
            
            if key not in combined_songs:
                combined_songs[key] = song
        return list(combined_songs.values())
    
    def save_to_csv(self, songs: List[Dict[str, str]], output_path: str) -> None:
        """Save the combined songs data to a CSV file."""
        if not songs:
            print("No songs to save.")
            return
        
        fieldnames = songs[0].keys()
        with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for song in songs:
                writer.writerow(song)
        print(f"Data saved to {output_path}")

    def remove_duplicates(self, songs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate songs based on artist, title, and spotify_id."""
        unique_songs = { (song['title']): song for song in songs }
        return list(unique_songs.values())
    
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    all_songs = preprocessor.load_all_data()
    all_songs = preprocessor.remove_duplicates(all_songs)
    print(f"Total unique songs loaded: {len(all_songs)}")
    preprocessor.save_to_csv(all_songs, "combined_songs.csv")

