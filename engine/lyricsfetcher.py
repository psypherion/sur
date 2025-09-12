import json
from titlenartists import TitleArtistExtractor
from lyrics import LyricsFetcher

def main(output_file="songs_with_lyrics.json"):
    extractor = TitleArtistExtractor()
    fetcher = LyricsFetcher()

    pairs = extractor.title_artist_pairs()  

    results = []
    for artist, title in pairs:
        print(f"Fetching lyrics: '{title}' by {artist} ...")
        lyrics = fetcher.fetch_lyrics(title, artist)
        results.append({
            "artist": artist,
            "title": title,
            "lyrics": lyrics
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} songs with lyrics to '{output_file}'.")

if __name__ == "__main__":
    main()
