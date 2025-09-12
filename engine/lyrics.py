import os
import lyricsgenius
from dotenv import load_dotenv

load_dotenv()

class _GeniusClient:
    def __init__(self) -> None:
        self.GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_CLIENT_ACCESS_TOKEN")

class LyricsFetcher:
    def __init__(self) -> None:
        client = _GeniusClient()
        access_token = client.GENIUS_ACCESS_TOKEN
        if not access_token:
            raise ValueError("Genius access token not found in environment variables.")
        self.genius = lyricsgenius.Genius(
            access_token,
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)", "(Acoustic)", "(Instrumental)", "(Cover)"],
            remove_section_headers=True,
            timeout=15
        )

    def fetch_lyrics(self, title: str, artist: str) -> str:
        try:
            song = self.genius.search_song(title=title, artist=artist)
            if song and song.lyrics:
                return song.lyrics
        except Exception as e:
            print(f"[GENIUS API ERROR] {e}")
        return "LYRICS NOT FOUND"

__all__ = ["LyricsFetcher"]

if __name__ == "__main__":
    fetcher = LyricsFetcher()
    title = "Blinding Lights"
    artist = "The Weeknd"
    lyrics = fetcher.fetch_lyrics(title, artist)
    print(lyrics)