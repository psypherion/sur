from dotenv import load_dotenv
import os
import base64
import requests
import json
import urllib.request
import re
import sys
import yt_dlp
from pydub import AudioSegment
import time
import traceback # Import traceback for better error logging

load_dotenv()

if "downloads" not in os.listdir():
    os.mkdir("downloads")

class SpotifyDownloader:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = self.get_token()

    def get_token(self):
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_encode = auth_string.encode("utf-8")
        auth_based64 = str(base64.b64encode(auth_encode), "utf-8")

        # Correct Spotify API token URL
        # Using googleusercontent.com as a placeholder as per user's original code
        url = 'https://accounts.spotify.com/api/token'
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "Basic " + auth_based64
        }
        data = {"grant_type": "client_credentials"}

        try:
            result = requests.post(url, headers=headers, data=data)
            result.raise_for_status() # Raise an exception for bad status codes
            json_data = json.loads(result.content)
            return json_data["access_token"]
        except requests.exceptions.RequestException as e:
            print(f"Error getting Spotify token: {e}")
            sys.exit(1)


class PlaylistDownloader:
    def __init__(self, token, playlist_link, resolution="144p"):
        self.token = token
        self.playlist_link = playlist_link
        self.resolution = resolution

    def get_playlist_id(self):
        try:
            id_si = self.playlist_link.split('playlist/')
            if len(id_si) < 2:
                 raise ValueError("Invalid playlist link format")
            playlist_part = id_si[1]
            if '?si' in playlist_part:
                return playlist_part.split('?')[0]
            else:
                return playlist_part
        except ValueError as e:
            print(f"Error parsing playlist ID from link: {e}")
            sys.exit(1)


    def get_playlist_details(self):
        playlist_id = self.get_playlist_id()
        # Correct Spotify API URL for playlists
        # Using googleusercontent.com as a placeholder as per user's original code
        base_url = "https://api.spotify.com/v1/playlists/"
        playlist_url = base_url + playlist_id
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            result = requests.get(playlist_url, headers=headers)
            result.raise_for_status() # Raise an exception for bad status codes
            return json.loads(result.content)
        except requests.exceptions.RequestException as e:
            print(f"Error getting playlist details from Spotify: {e}")
            sys.exit(1)


    def extract_song_artist_name(self, playlist_dict):
        # --- Original Extraction Logic (as per your debug output) ---
        count_num = 0
        arr = []
        song_name = []
        artist_name = []
        name_parent = []

        try:
            # This logic iterates through keys of playlist_dict["tracks"] which seems unusual
            # but is kept as it's the one producing your debug output and claims to work.
            for item in playlist_dict["tracks"]:
                count_num += 1
                arr.append(playlist_dict["tracks"][item])
                # Assuming arr[1] contains the list of track items based on your logic
                if count_num > 1:
                    # Check if arr[1] is actually a list before iterating
                    if isinstance(arr[1], list):
                        for track_item in arr[1]:
                            # Check if 'track' key exists and is not None
                            if "track" in track_item and track_item["track"]:
                                track = track_item["track"]
                                # Check if 'name' key exists before accessing
                                if "name" in track and track["name"] not in song_name:
                                    song_name.append(track["name"])
                                    # Extract artist names from album section as per your logic
                                    if "album" in track and isinstance(track["album"], dict):
                                         # Use .values() to iterate over album details
                                         for album_items in track["album"].values():
                                             name_parent.append(album_items)

            # Extract artist names from the collected name_parent list
            for item in name_parent:
                # Check if item is a list, not empty, and contains a dictionary with a 'name' key at index 0
                if isinstance(item, list) and item and isinstance(item[0], dict) and "name" in item[0]:
                    artist_name.append(item[0]["name"])
                # Add a placeholder if artist extraction fails for a corresponding song
                # This simple check might not perfectly align artists if the extraction is complex
                # print(f"Debug: Artist extraction failed for item: {item}") # Optional debug


            print("Extracted (original logic):", song_name, artist_name) # Debugging print

            # Ensure lists are of the same length by trimming the longer one
            # This might misalign songs and artists if the extraction logic is flawed.
            # A better approach would build song/artist pairs during the first loop.
            min_len = min(len(song_name), len(artist_name))
            song_name = song_name[:min_len]
            artist_name = artist_name[:min_len]

        except Exception as e:
            print(f"Error extracting song and artist names using original logic: {e}")
            traceback.print_exc()
            # Return empty lists on failure to prevent further errors
            return [], []


        # --- Standard Extraction Logic (commented out) ---
        # song_name_standard = []
        # artist_name_standard = []
        # try:
        #     if "tracks" in playlist_dict and "items" in playlist_dict["tracks"]:
        #         for item in playlist_dict["tracks"]["items"]:
        #             if "track" in item and item["track"]:
        #                 track = item["track"]
        #                 if "name" in track:
        #                     song_name_standard.append(track["name"])
        #                 else:
        #                     song_name_standard.append("Unknown Song")
        #
        #                 artists = track.get("artists")
        #                 if artists and isinstance(artists, list) and len(artists) > 0 and "name" in artists[0]:
        #                      artist_name_standard.append(artists[0]["name"])
        #                 else:
        #                      artist_name_standard.append("Unknown Artist")
        # except Exception as e:
        #     print(f"Error extracting song and artist names using standard logic: {e}")
        #     traceback.print_exc()


        return song_name, artist_name


    def generate_youtube_queries(self, song_name, artist_name):
        # Ensure song_name and artist_name are lists of the same length
        if len(song_name) != len(artist_name):
            print("Warning: Mismatch in song and artist list lengths. Skipping query generation.")
            return []

        search_queries_text = [f"{song_name[i]} by {artist_name[i]}" for i in range(len(artist_name))]

        youtube_urls = []
        print("\nGenerating YouTube URLs...")
        for i, query_text in enumerate(search_queries_text):
            search_query = query_text.replace(" ", "+")
            # Using googleusercontent.com as a placeholder as per user's original code
            search_url = "https://www.youtube.com/results?search_query=" + search_query
            try:
                print(f"Searching YouTube for: {query_text}...")
                # Using requests instead of urllib.request for potentially better handling
                response = requests.get(search_url)
                response.raise_for_status() # Raise an exception for bad status codes
                html = response.text

                # Use a more robust regex pattern for finding video IDs
                video_ids = re.findall(r'/watch\?v=([a-zA-Z0-9_-]{11})', html)

                if video_ids:
                    # Using googleusercontent.com as a placeholder as per user's original code
                    video_url = "https://www.youtube.com/watch?v=" + video_ids[0]
                    youtube_urls.append(video_url)
                    print(f"Found: {video_url}")
                else:
                    print(f"No video found for query: {query_text}")
            except requests.exceptions.RequestException as e:
                print(f"Error searching YouTube for '{query_text}': {e}")
            except Exception as e:
                print(f"An unexpected error occurred during Youtube for '{query_text}': {e}")
                traceback.print_exc()


        return youtube_urls

class AudioDownloader:
    def __init__(self, output_directory='downloads', desired_bitrate="192k", resolution="144p"):
        self.output_directory = output_directory
        self.desired_bitrate = desired_bitrate
        self.resolution = resolution
        self.total_time = 0

    def sanitize_filename(self, title):
        # Remove characters that are not allowed in filenames across common OS
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', title) # Replace invalid characters with underscore
        sanitized = re.sub(r'\s+', ' ', sanitized).strip() # Replace multiple spaces with single space and strip leading/trailing
        return sanitized

    # --- Modified download_audio based on your original working code ---
    def download_audio(self, video_urls, song_name_list, artist_name_list, playlist_name):
        # List to store details of successfully downloaded songs
        downloaded_songs_details = []

        print(f"\nStarting audio downloads for playlist: {playlist_name}")

        # Fix path joining for output directory
        output_directory_path = os.path.join(self.output_directory, self.sanitize_filename(playlist_name))

        if not os.path.exists(output_directory_path):
            os.makedirs(output_directory_path)
            print(f"Created directory: {output_directory_path}")

        # Write playlist details to a text file (optional, keeps original behavior)
        playlist_details_file = os.path.join(output_directory_path, f"{self.sanitize_filename(playlist_name)}.txt")
        with open(playlist_details_file, "w", encoding="utf-8") as playlist_name_file: # Use 'w' to overwrite each time
            playlist_name_file.write(f"Playlist Name: {playlist_name}\n\n")
            # Assuming song_name_list and artist_name_list are aligned
            for i in range(min(len(song_name_list), len(artist_name_list))):
                 playlist_name_file.write(f"Title: {song_name_list[i]}\nArtist: {artist_name_list[i]}\n\n")
        print(f"Playlist song list saved to {playlist_details_file}")


        # Iterate through the video URLs, using the index to match with song/artist names
        for i, url in enumerate(video_urls):
            # Ensure index is within bounds of song/artist lists
            if i >= len(song_name_list) or i >= len(artist_name_list):
                print(f"Skipping video URL {url} due to missing song/artist information.")
                continue # Skip to the next URL

            current_song_name = song_name_list[i]
            current_artist_name = artist_name_list[i]
            unique_id = i + 1 # Simple 1-based index as unique identifier

            try:
                start_time = time.time()
                print(f"\n[{unique_id}/{len(video_urls)}] Downloading and converting: {current_song_name} by {current_artist_name}")
                print(f"Source URL: {url}")

                # --- Original yt-dlp options from your working code ---
                ydl_opts = {
                    'format': f'bestvideo[height={self.resolution}]+bestaudio/best', # Download best video at resolution + best audio
                    'outtmpl': f'{output_directory_path}/%(title)s.%(ext)s', # Output template for downloaded file (e.g., .mp4)
                    'quiet': True,
                    'no_warnings': True,
                    'extractor_args': {
                        'youtube': {
                            'quiet': True,
                            'nocheckcertificate': True,
                            'source_address': '0.0.0.0',
                        },
                        'youtube:info': {
                            'skip': ['ios', 'android', 'm3u8'],
                        },
                    },
                    # No postprocessors here, as pydub does the conversion
                }

                info_dict = None
                downloaded_file_path = None # Variable to store the path of the downloaded video/audio file

                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        # --- Original method to download and get info ---
                        info_dict = ydl.extract_info(url, download=True)
                        video_title = info_dict.get('title', current_song_name) # Use extracted title or fallback
                        sanitized_title = self.sanitize_filename(video_title)

                        # --- Original method to get the path of the downloaded file ---
                        # This relies on ydl.prepare_filename after download=True
                        # Note: This might not work reliably in all yt-dlp versions/scenarios.
                        # If issues persist, the hook method from previous attempts is more robust.
                        downloaded_file_path = ydl.prepare_filename(info_dict)


                    # Check if the download was successful and the file exists
                    if downloaded_file_path and os.path.exists(downloaded_file_path):
                        print(f"Successfully downloaded raw file: '{downloaded_file_path}'")
                        # --- Original pydub conversion logic ---
                        try:
                            audio = AudioSegment.from_file(downloaded_file_path)
                            output_mp3 = os.path.join(output_directory_path, f"{sanitized_title}.mp3")
                            # Check if output directory exists before saving (redundant with outer check, but safe)
                            if not os.path.exists(os.path.dirname(output_mp3)):
                                os.makedirs(os.path.dirname(output_mp3))

                            audio.export(output_mp3, format="mp3", bitrate=self.desired_bitrate)

                            # --- Add successful download details to the list ---
                            downloaded_songs_details.append({
                                "id": unique_id,
                                "spotify_song_name": current_song_name,
                                "spotify_artist_name": current_artist_name,
                                "youtube_title": video_title, # Include YouTube title for reference
                                "file_path": output_mp3 # Store the final MP3 path
                            })
                            print(f"Successfully converted to MP3 and saved as '{output_mp3}'")

                        except FileNotFoundError:
                             print(f"Error: raw downloaded file not found at '{downloaded_file_path}' for pydub conversion.")
                        except Exception as e:
                             print(f"Error converting '{downloaded_file_path}' to MP3: {e}")
                             traceback.print_exc()
                    else:
                         print(f"Error: yt-dlp download failed for URL: {url}. Cannot proceed with conversion.")

                except yt_dlp.utils.DownloadError as e:
                    print(f"Warning: yt-dlp download error for '{current_song_name}' ({url}): {e}")
                    if 'Video unavailable' in str(e):
                        print(f"Reason: Video is unavailable (likely due to copyright or region restriction).")
                    # traceback.print_exc() # Optional: Print full traceback for debug
                except Exception as e:
                    # Catch any other unexpected errors during the yt-dlp process itself
                    print(f"An unexpected error occurred during yt-dlp process for '{current_song_name}' ({url}): {e}")
                    traceback.print_exc()


                end_time = time.time()
                elapsed_time = end_time - start_time
                self.total_time += elapsed_time
                print(f"Time taken for this song: {elapsed_time:.2f} seconds")

            except Exception as e:
                # Catch any errors outside the core yt-dlp process loop
                print(f"An error occurred processing URL {url} (outside yt-dlp call): {e}")
                traceback.print_exc()

        print(f"\nTotal time spent on downloads: {self.total_time:.2f} seconds")
        # --- Return the collected details list ---
        return downloaded_songs_details

    def delete_video_files(self, folder_path):
        print("\nDeleting intermediate video files...")
        deleted_count = 0
        try:
            # Ensure the folder exists before listing
            if not os.path.exists(folder_path):
                print(f"Folder not found for deletion: {folder_path}")
                return

            files = os.listdir(folder_path)
            for file in files:
                # Check for common video extensions, be careful not to delete the .mp3 files
                if file.lower().endswith((".mp4", ".webm", ".mkv", ".flv", ".avi")): # Use .lower() for case-insensitivity
                    file_path = os.path.join(folder_path, file)
                    try:
                        # Add a check to ensure it's actually a file before attempting deletion
                        if os.path.isfile(file_path):
                             os.remove(file_path)
                             # print(f"Deleted: '{file}'") # Commented out to reduce output noise during cleanup
                             deleted_count += 1
                        # else: print(f"Skipping '{file}' as it's not a file.") # Optional debug
                    except OSError as e:
                         print(f"Error deleting file '{file}': {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred deleting file '{file}': {e}")
                        traceback.print_exc()
        except Exception as e:
            print(f"An error occurred during video file deletion: {e}")
            traceback.print_exc()

        if deleted_count > 0:
            print(f"Finished deleting {deleted_count} intermediate video files.")
        else:
             print("No intermediate video files found to delete.")


class SpotifyPlaylistProcessor:
    def __init__(self, client_id, client_secret, playlist_link, resolution="144p"):
        self.spotify_downloader = SpotifyDownloader(client_id, client_secret)
        self.playlist_downloader = PlaylistDownloader(self.spotify_downloader.token, playlist_link, resolution)
        # Pass resolution to AudioDownloader
        self.audio_downloader = AudioDownloader(resolution=resolution)

    def process_playlist(self):
        print("Fetching playlist details from Spotify...")
        # Get playlist details
        playlist_dict = self.playlist_downloader.get_playlist_details()
        playlist_name = playlist_dict.get("name", "Downloaded_Playlist") # Get playlist name safely

        print(f"Processing playlist: {playlist_name}")

        # Extract playlist name, song names, and artist names
        # This will use the original extraction logic kept in PlaylistDownloader
        song_name_list, artist_name_list = self.playlist_downloader.extract_song_artist_name(playlist_dict)

        if not song_name_list:
            print("No songs found in the playlist or failed to extract song details. Exiting.")
            return # Exit if no songs are found


        print(f"\nFound {len(song_name_list)} songs in the playlist.")


        # Generate YouTube queries and get URLs
        youtube_urls = self.playlist_downloader.generate_youtube_queries(song_name_list, artist_name_list)

        # Note: generate_youtube_queries might return fewer URLs than there are songs
        # Filter song_name_list and artist_name_list to match the found youtube_urls
        # This ensures the lists passed to download_audio are aligned with the URLs it will process.
        # The current download_audio loop handles index mismatches with a 'continue',
        # but aligning lists here makes the unique_id correspond to the *search result* index.
        # Keeping the current download_audio loop logic for now as it matches the provided code structure.


        if not youtube_urls:
            print("No matching YouTube videos found for any songs. Exiting.")
            return # Exit if no YouTube URLs are found

        print(f"\nFound {len(youtube_urls)} matching YouTube videos.")

        # Define output directory path (used by AudioDownloader, but defined here for clarity)
        output_directory_path = os.path.join(self.audio_downloader.output_directory, self.audio_downloader.sanitize_filename(playlist_name))

        # Create the output directory if it doesn't exist (handled in AudioDownloader, but safe to check/create here too)
        if not os.path.exists(output_directory_path):
            os.makedirs(output_directory_path)


        # Download audio from YouTube and collect details
        # Pass the full song/artist lists. The download_audio loop will handle index alignment.
        downloaded_songs_details = self.audio_downloader.download_audio(
            youtube_urls,
            song_name_list=song_name_list, # Pass original lists
            artist_name_list=artist_name_list, # Pass original lists
            playlist_name=playlist_name # Pass playlist name for directory creation
        )

        # Save song details to a JSON file
        json_file_path = os.path.join(output_directory_path, "song_details.json")
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(downloaded_songs_details, f, indent=4, ensure_ascii=False)
            print(f"\nSuccessfully saved song details to '{json_file_path}'")
        except Exception as e:
            print(f"Error saving song details to JSON: {e}")
            traceback.print_exc()


        # Delete intermediate video files after all downloads are attempted
        self.audio_downloader.delete_video_files(output_directory_path)


# Example usage:

# Check if client ID and secret are provided in environment variables
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

if not client_id or not client_secret:
    print("Error: Spotify CLIENT_ID and CLIENT_SECRET environment variables not set.")
    print("Please create a .env file with these variables or set them in your environment.")
    sys.exit(1)


# Check if playlist link is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python your_script_name.py <playlist_link>")
    sys.exit(1)

playlist_link = sys.argv[1]

# Create a SpotifyPlaylistProcessor instance
processor = SpotifyPlaylistProcessor(client_id, client_secret, playlist_link)

# Process the playlist
processor.process_playlist()