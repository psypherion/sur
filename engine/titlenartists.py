import os
from typing import List
import json

class _GenresClient:
    def __init__(self) -> None:
        pass

    def files(self) -> List[str]:
        return [os.path.join(os.getcwd(),"db/", f) 
                for f in os.listdir("db/") if f.endswith(".json")]

class TitleArtistExtractor:
    def __init__(self) -> None:
        self.client = _GenresClient()
        self.files = self.client.files()
        if not self.files:
            raise ValueError("No JSON files found in the 'db/' directory.")
        
    def title_artist_pairs(self) -> List[tuple]:
        song_artists = set()
        for file in self.files:
            content = json.load(open(file, "r"))
            for song in content:
                if "artist" in song and "title" in song:
                    artist: str = song["artist"]
                    title: str = song["title"]
                    song_artists.add((artist, title))
        return list(song_artists)

__all__ = ["TitleArtistExtractor"]

if __name__ == "__main__":
    extractor = TitleArtistExtractor()
    pairs = extractor.title_artist_pairs()
    for artist, title in pairs:
        print(f"Artist: {artist}, Title: {title}")
    