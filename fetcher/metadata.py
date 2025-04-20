from dotenv import load_dotenv
import os
import base64
import requests
import json
import sys

load_dotenv()

class SpotifyDownloader:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = self.get_token()

    def get_token(self):
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_encode = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")

        url = 'https://accounts.spotify.com/api/token'
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {auth_encode}"
        }
        data = {"grant_type": "client_credentials"}

        result = requests.post(url, headers=headers, data=data)
        result.raise_for_status()
        return result.json()["access_token"]

class PlaylistDownloader:
    def __init__(self, token, playlist_link):
        self.token = token
        self.playlist_link = playlist_link

    def get_playlist_id(self):
        if 'playlist/' not in self.playlist_link:
            raise ValueError("Invalid playlist link")
        return self.playlist_link.split('playlist/')[1].split('?')[0]

    def get_playlist_details(self):
        playlist_id = self.get_playlist_id()
        url = f"https://api.spotify.com/v1/playlists/{playlist_id}"
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def extract_song_artist_names(self, playlist_dict):
        songs = []
        artists = []
        for item in playlist_dict["tracks"]["items"]:
            track = item.get("track", {})
            song_name = track.get("name")
            artist_info = track.get("artists", [{}])[0]
            artist_name = artist_info.get("name")
            if song_name and artist_name:
                songs.append(song_name)
                artists.append(artist_name)
        return songs, artists

class SpotifyPlaylistProcessor:
    def __init__(self, client_id, client_secret, playlist_link):
        self.spotify_downloader = SpotifyDownloader(client_id, client_secret)
        self.playlist_downloader = PlaylistDownloader(self.spotify_downloader.token, playlist_link)

    def process_playlist(self):
        playlist_dict = self.playlist_downloader.get_playlist_details()
        playlist_name = playlist_dict.get("name")
        song_names, artist_names = self.playlist_downloader.extract_song_artist_names(playlist_dict)
        print(f"Playlist Name: {playlist_name}")
        print(f"Song Names: {song_names}")
        print(f"Artist Names: {artist_names}")
        return playlist_dict, song_names, artist_names

class AudioAnalysis:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = self.get_token()

    def get_token(self):
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_encoded = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
        url = 'https://accounts.spotify.com/api/token'
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {auth_encoded}"
        }
        data = {"grant_type": "client_credentials"}

        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()["access_token"]

    def get_audio_analysis(self, track_id):
        url = f"https://api.spotify.com/v1/audio-analysis/{track_id}"
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def extract_audio_features(self, audio_analysis_dict, save_to_file=False, filename=None):
        track_data = audio_analysis_dict.get("track", {})
        audio_features = {
            "tempo": track_data.get("tempo"),
            "key": track_data.get("key"),
            "mode": track_data.get("mode"),
            "time_signature": track_data.get("time_signature"),
            "duration_ms": track_data.get("duration"),
            "loudness": track_data.get("loudness")
        }
        if save_to_file and filename:
            with open(filename, 'w') as f:
                json.dump(audio_features, f, indent=2)
            print(f"Audio features saved to {filename}")
        return audio_features

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python downloader.py <playlist_link>")
        sys.exit(1)

    playlist_link = sys.argv[1]
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")

    processor = SpotifyPlaylistProcessor(client_id, client_secret, playlist_link)
    playlist_details, song_names, artist_names = processor.process_playlist()

    with open('playlist_details.json', 'w') as f:
        json.dump(playlist_details, f, indent=2)
    print("Playlist details saved to playlist_details.json")

    track_links = [item["track"]["external_urls"]["spotify"] for item in playlist_details["tracks"]["items"]]
    with open('track_links.json', 'w') as f:
        json.dump(track_links, f, indent=2)
    print("Track links saved to track_links.json")

    audio_analyzer = AudioAnalysis(client_id, client_secret)
    for track_link in track_links:
        track_id = track_link.split("/")[-1]
        try:
            analysis = audio_analyzer.get_audio_analysis(track_id)
            features = audio_analyzer.extract_audio_features(analysis, save_to_file=True, filename=f"audio_features_{track_id}.json")
        except Exception as e:
            print(f"Error processing {track_link}: {e}")