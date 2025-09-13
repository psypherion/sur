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

class _TrebiLoader:
    """Loads configuration and environment variables for data processing."""
    def __init__(self) -> None:
        self.TREBI_CSV = os.getenv("TREBI_CSV")
        if not self.TREBI_CSV:
            raise ValueError("TREBI_CSV path not found in environment variables.")
        self.csv_path = os.path.join(os.getcwd(), self.TREBI_CSV)

class TrebiPreprocessor:
    """Processes the Trebi dataset from a CSV file with mixed delimiters."""
    def __init__(self) -> None:
        loader = _TrebiLoader()
        self.csv_path: str = loader.csv_path

    def __load_songs(self) -> List[Dict[str, str]]:
        """Load songs from Trebi CSV file with comma header and semicolon data."""
        if not os.path.isfile(self.csv_path):
            logging.error(f"CSV file not found: {self.csv_path}")
            return []
        
        songs = []
        with open(self.csv_path, mode='r', encoding='utf-8') as csvfile:
            header_line = csvfile.readline().strip()
            fieldnames = [h.strip('"') for h in header_line.split(",")]

            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                clean_row = [value.strip('"') for value in row]
                row_dict = dict(zip(fieldnames, clean_row))
                song = {
                    "spotify_id": row_dict.get("spotify_id", "").strip(),
                    "title": row_dict.get("name", "").strip(),
                    "artist": row_dict.get("artist", "").strip(),
                }
                songs.append(song)
        
        logging.info(f"Total songs loaded from Trebi dataset: {len(songs)}")
        return songs
    
    def get_songs(self) -> List[Dict[str, str]]:
        """Public method to get the list of songs."""
        return self.__load_songs()
    
if __name__ == "__main__":
    processor = TrebiPreprocessor()
    songs = processor.get_songs()
    print(f"Loaded {len(songs)} songs from Trebi dataset.")
    print(f"First 5 songs: {songs[:5]}")