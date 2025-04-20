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

        url = 'https://accounts.spotify.com/api/token'
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "Basic " + auth_based64
        }
        data = {"grant_type": "client_credentials"}

        result = requests.post(url, headers=headers, data=data)
        json_data = json.loads(result.content)
        return json_data["access_token"]

class PlaylistDownloader:
    def __init__(self, token, playlist_link, resolution="144p"):
        self.token = token
        self.playlist_link = playlist_link
        self.resolution = resolution

    def get_playlist_id(self):
        id_si = self.playlist_link.split('playlist/')
        if '?si' in id_si[1]:
            return id_si[1].split('?')[0]
        else:
            return id_si[1]

    def get_playlist_details(self):
        playlist_id = self.get_playlist_id()
        base_url = "https://api.spotify.com/v1/playlists/"
        playlist_url = base_url + playlist_id
        headers = {"Authorization": f"Bearer {self.token}"}
        result = requests.get(playlist_url, headers=headers)
        return json.loads(result.content)

    def extract_song_artist_name(self, playlist_dict):
        count_num = 0
        arr = []
        num = 0
        name_parent = []
        artist_name = []
        song_name = []

        for item in playlist_dict["tracks"]:
            count_num += 1
            arr.append(playlist_dict["tracks"][item])
            if count_num > 1:
                for i in range(0, len(arr[1])):
                    for keys in arr[1][i]["track"]:
                        if arr[1][i]["track"]["name"] not in song_name:
                            song_name.append(arr[1][i]["track"]["name"])
                            for album_items in arr[1][i]["track"]["album"]:
                                name_parent.append(arr[1][i]["track"]["album"][album_items])

        for item in name_parent:
            if isinstance(item, list) and item and "name" in item[0]:
                artist_name.append(item[0]["name"])
        return song_name, artist_name

    def generate_youtube_queries(self, song_name, artist_name):
        search_query = [f"{song_name[i]} by {artist_name[i]}" for i in range(len(artist_name))]
        search_queries = [query.replace(" ", "+") for query in search_query]

        youtube_queries = []
        for query in search_queries:
            try:
                html = urllib.request.urlopen("https://www.youtube.com/results?search_query="+query)
                video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
                youtube_queries.append("https://www.youtube.com/watch?v=" + video_ids[0])
            except Exception:
                pass

        return youtube_queries

class AudioDownloader:
    def __init__(self, output_directory='downloads', desired_bitrate="192k", resolution="144p"):
        self.output_directory = output_directory
        self.desired_bitrate = desired_bitrate
        self.resolution = resolution
        self.total_time = 0
    def sanitize_filename(self, title):
        return title.replace("/", "_")

    def download_audio(self, video_urls, song_name, artist_name, playlist_name):
        count = 0
        song_name = song_name
        artist_name = artist_name
        playlist_name = playlist_name
        output_directory = self.output_directory + "/" + playlist_name
        playlist_details_file = os.path.join(output_directory, f"{playlist_name}.txt")
        with open(playlist_details_file, "a", encoding="utf-8") as playlist_name_file:
            playlist_name_file.write(f"Playlist Name: {playlist_name}\n\n")
        for song in song_name:
            with open(playlist_details_file, "a", encoding="utf-8") as song_details_file:
                song_details_file.write(f"Title: {song_name[count]}\nArtist: {artist_name[count]}\n\n")
            count += 1
        print(f"\nPlaylist Details:\nSong Name: {song_name}\nArtist Name: {artist_name}\n")
        for url in video_urls:
            try:
                start_time = time.time()
                ydl_opts = {
                    'format': f'bestvideo[height={self.resolution}]+bestaudio/best',
                    'outtmpl': f'{output_directory}/%(title)s.%(ext)s',
                    'quiet': True,
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
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=True)
                    video_title = info_dict.get('title', 'video')
                    sanitized_title = self.sanitize_filename(video_title)
                    audio_file_path = ydl.prepare_filename(info_dict)

                    audio = AudioSegment.from_file(audio_file_path)

                    if not os.path.exists(output_directory):
                        os.makedirs(output_directory)

                    output_mp3 = os.path.join(output_directory, f"{sanitized_title}.mp3")

                    audio.export(output_mp3, format="mp3", bitrate=self.desired_bitrate)
                    count += 1
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    self.total_time += elapsed_time

                    print(f"Video '{video_title}' converted to MP3 and saved as '{output_mp3}'")
                    print(f"Time taken: {elapsed_time} seconds\n")

            except yt_dlp.utils.DownloadError as e:
                if 'Video unavailable' in str(e):
                    print(f"Warning: Video '{url}' is unavailable due to copyright restrictions.")
                else:
                    print(f"Error: {e}")

    def delete_video_files(self, folder_path):
        files = os.listdir(folder_path)
        for file in files:
            if file.endswith(".mp4"):
                file_path = os.path.join(folder_path, file)
                try:
                    os.remove(file_path)
                    print(f"File '{file}' deleted successfully.")
                except Exception as e:
                    print(f"Error deleting file '{file}': {e}")

class SpotifyPlaylistProcessor:
    def __init__(self, client_id, client_secret, playlist_link, resolution="144p"):
        self.spotify_downloader = SpotifyDownloader(client_id, client_secret)
        self.playlist_downloader = PlaylistDownloader(self.spotify_downloader.token, playlist_link, resolution)
        self.audio_downloader = AudioDownloader()

    def process_playlist(self):
        # Get playlist details
        playlist_dict = self.playlist_downloader.get_playlist_details()

        # Extract playlist name, song names, and artist names
        playlist_name = playlist_dict["name"]
        song_name, artist_name = self.playlist_downloader.extract_song_artist_name(playlist_dict)

        # Generate YouTube search queries
        youtube_queries = self.playlist_downloader.generate_youtube_queries(song_name, artist_name)

        # Define output directory path
        output_directory_path = f'{self.audio_downloader.output_directory}/{playlist_name}'

        # Create the output directory if it doesn't exist
        if output_directory_path not in os.listdir():
            os.mkdir(output_directory_path)

        # Download audio from YouTube
        self.audio_downloader.download_audio(
            youtube_queries,
            song_name=song_name,
            artist_name=artist_name,
            playlist_name=playlist_name
        )
        self.audio_downloader.delete_video_files(output_directory_path)

# Example usage:

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
playlist_link = sys.argv[1]
if len(sys.argv) != 2:
    print("Usage: python downloader.py <playlist_link>")
    sys.exit(1)
# Create a SpotifyPlaylistProcessor instance
processor = SpotifyPlaylistProcessor(client_id, client_secret, playlist_link)

# Process the playlist
processor.process_playlist()


import re
import os
import json
import random
import logging
import asyncio
import requests
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async

# === Configuration ===
BASE_URL = "https://www.azlyrics.com/lyrics/"
INPUT_FILE = "fetcher/downloads/Rock playlist/Rock playlist.txt"
OUTPUT_DIR = "fetcher/downloads/Rock playlist/lyrics"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def sanitize(text: str) -> str:
    text = re.sub(r'(?i)^\s*the\s+', '', text)
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    text = re.sub(r'\s+', '', text)
    text = re.sub(r"['\"]", '', text)  # remove single and double quotes
    text = re.sub(r'\bremastered\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bremaster\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bremix\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bversion\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfeat\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bft\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'-', '', text)  # remove hyphens
    text = re.sub(r'\b\d{4}\b', '', text)  # remove years (4 digits)
    text = re.sub(r'\s+', '', text)  # remove tabs
    text = re.sub(r'\s+', ' ', text)  # remove multiple spaces
    text = re.sub(r'\s*$', '', text)  # remove trailing spaces
    text = re.sub(r'^\s*', '', text)  # remove leading spaces
    text = re.sub(r'\.$', '', text)  # remove dot from the last position
    return text.strip()


def construct_url(artist: str, song: str) -> str:
    return f"{BASE_URL}{sanitize(artist.lower())}/{sanitize(song.lower())}.html"


def parse_song_artist(filepath: str) -> list:
    songs = []
    song = artist = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Title:"):
                song = line.split(":", 1)[1].strip()
            elif line.startswith("Artist:"):
                artist = line.split(":", 1)[1].strip()
                if song and artist:
                    songs.append((song, artist))
                    song = artist = None
    return songs


def fallback_fetch(url: str) -> str:
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            match = re.search(
                r"<!--\s*Usage of azlyrics\.com content.*?-->(.*?)<script>",
                resp.text, re.DOTALL | re.IGNORECASE
            )
            return match.group(1).strip() if match else None
    except Exception:
        return None


async def fetch_lyrics_playwright(page, url: str) -> str:
    for attempt in range(3):
        try:
            await page.goto(url, timeout=15000)
            html = await page.content()
            match = re.search(
                r"<!--\s*Usage of azlyrics\.com content.*?-->(.*?)<script>",
                html, re.DOTALL | re.IGNORECASE
            )
            return match.group(1).strip() if match else None
        except Exception as e:
            logging.warning(f"[Retry {attempt+1}] Page.goto failed for {url}: {e}")
            await asyncio.sleep(random.uniform(3, 6))
    return None


async def main():
    logging.basicConfig(level=logging.INFO)

    songs = parse_song_artist(INPUT_FILE)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await stealth_async(page)

        for song, artist in songs:
            url = construct_url(artist, song)
            logging.info(f"Fetching: {artist} - {song} → {url}")

            await page.set_extra_http_headers({
                "User-Agent": random.choice(USER_AGENTS),
                "Accept-Language": "en-US,en;q=0.9",
            })
            await page.set_viewport_size({
                "width": random.choice([1280, 1366, 1440, 1920]),
                "height": random.choice([720, 768, 900, 1080])
            })
            await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => false});")

            await asyncio.sleep(random.uniform(4, 10))

            lyrics = await fetch_lyrics_playwright(page, url)

            if not lyrics:
                logging.warning(f"Playwright failed for {artist} - {song}. Trying fallback...")
                lyrics = fallback_fetch(url)

            if lyrics:
                filename = f"{OUTPUT_DIR}/{sanitize(artist)} - {sanitize(song)}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(lyrics)
                logging.info(f"✔ Saved: {filename}")
            else:
                logging.error(f"✘ Lyrics not found for {artist} - {song}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())


import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import json
import logging
import urllib.parse
import re # For cleaning header keys

# --- Configuration ---
INPUT_PLAYLIST_URL = "https://open.spotify.com/playlist/37i9dQZF1DWXRqgorJj26U" # Example: Spotify "lofi beats"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Function to create the direct songdata.io search URL ---
def create_songdata_search_url(playlist_url: str) -> str:
    """
    Creates the direct search URL for songdata.io by URL-encoding the playlist link.
    """
    base_url = "https://songdata.io/search?query="
    encoded_playlist_url = urllib.parse.quote(playlist_url, safe='')
    direct_url = base_url + encoded_playlist_url
    logging.info(f"Constructed direct URL: {direct_url}")
    return direct_url

# --- Function to clean header text for JSON keys ---
def clean_header_key(header_text):
    """Cleans header text to be a valid and readable JSON key."""
    if not header_text:
        return "unknown_header"
    # Replace '#' with 'rank', remove special chars, lowercase, replace spaces with underscores
    key = header_text.strip().lower()
    key = key.replace('#', 'rank')
    key = re.sub(r'[^\w\s-]', '', key) # Remove non-alphanumeric (excluding space, underscore, hyphen)
    key = re.sub(r'[-\s]+', '_', key) # Replace spaces/hyphens with underscores
    return key or "unknown_header"


# --- Playwright Scraping Function (Updated for Settings Interaction) ---
async def extract_songdata_playlist_direct(direct_search_url: str):
    """
    Navigates directly to the songdata.io search results page,
    opens the settings dropdown, checks all checkboxes,
    and scrapes the resulting tracklist table.
    """
    logging.info(f"Starting direct playlist extraction from: {direct_search_url}")
    results = {"data": None, "error": None}
    browser = None
    context = None
    page = None

    try:
        async with async_playwright() as p:
            # Use headless=False for debugging the interaction
            browser = await p.chromium.launch(headless=True, timeout=60000)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
            )
            page = await context.new_page()

            logging.info(f"Navigating directly to results page: {direct_search_url}...")
            await page.goto(direct_search_url, timeout=120000, wait_until='networkidle')

            # --- Wait for Initial Table ---
            results_table_selector = "#table_chart" # Confirmed from original HTML
            logging.info(f"Waiting for results table selector: {results_table_selector}")
            try:
                await page.wait_for_selector(results_table_selector, state="visible", timeout=180000)
                logging.info("Initial results table found.")
            except PlaywrightTimeoutError:
                 logging.error(f"Timeout waiting for initial results table on direct URL.")
                 # Attempt to diagnose common issues
                 page_content = await page.content() # Get page content for diagnosis
                 if "rate limit" in page_content.lower():
                      results["error"] = "Rate limit likely hit on songdata.io."
                 elif "could not find playlist" in page_content.lower():
                      results["error"] = "Songdata.io reported: Could not find playlist."
                 else:
                     results["error"] = "Timeout waiting for results table. Playlist might be invalid/private, or site issues."
                 if browser: await browser.close()
                 return results


            # --- Interact with Settings/Columns Dropdown ---
            settings_button_selector = "button.settingbtn" # Confirmed from new HTML
            dropdown_menu_selector = "div#setting.setting-dropdown" # Confirmed from new HTML
            checkbox_selector = f"{dropdown_menu_selector} input[type='checkbox'].toggle-col" # Confirmed from new HTML

            try:
                logging.info(f"Looking for settings button: '{settings_button_selector}'")
                settings_button = page.locator(settings_button_selector).first
                await settings_button.wait_for(state="visible", timeout=15000)
                logging.info("Settings button found. Clicking...")
                await settings_button.click()

                logging.info(f"Waiting for dropdown menu: '{dropdown_menu_selector}'")
                dropdown_menu = page.locator(dropdown_menu_selector)
                await dropdown_menu.wait_for(state="visible", timeout=10000)
                logging.info("Dropdown menu is visible.")

                logging.info(f"Finding checkboxes: '{checkbox_selector}'")
                checkbox_locators = dropdown_menu.locator("input[type='checkbox'].toggle-col")
                count = await checkbox_locators.count()
                logging.info(f"Found {count} checkboxes in the settings dropdown.")

                checked_count = 0
                # Iterate through checkboxes and check if not already checked
                for i in range(count):
                    checkbox = checkbox_locators.nth(i)
                    checkbox_id = await checkbox.get_attribute('id')
                    label_locator = dropdown_menu.locator(f"label[for='{checkbox_id}']")
                    label_text = await label_locator.text_content() if await label_locator.count() > 0 else f"Label for {checkbox_id}"

                    is_checked = await checkbox.is_checked()
                    logging.debug(f"Checkbox {i+1} ('{label_text.strip()}'): Already checked = {is_checked}")
                    if not is_checked:
                        logging.info(f"Checking checkbox {i+1}: '{label_text.strip()}'")
                        # Using click on the label is often more robust for custom checkboxes
                        await label_locator.click(timeout=5000)
                        # Alternative: await checkbox.check(force=True, timeout=5000) # force=True might help if label click fails
                        checked_count += 1
                        await page.wait_for_timeout(50) # Small pause after checking

                logging.info(f"Finished checking boxes. {checked_count} boxes were newly checked.")

                # Click the settings button again to attempt to close the dropdown
                logging.info("Clicking settings button again to close dropdown...")
                await settings_button.click()
                await page.wait_for_timeout(500) # Wait a moment for UI updates

            except PlaywrightTimeoutError as e:
                logging.warning(f"Could not find or interact with the Settings dropdown: {e}")
                logging.warning("Proceeding to scrape table with currently visible columns.")
            except Exception as e:
                 logging.error(f"Unexpected error during settings interaction: {e}", exc_info=True)
                 logging.warning("Attempting to continue scraping despite settings error.")


            # --- Extract Data from Table ---
            # Re-fetch headers AFTER potential column changes
            logging.info("Fetching final table headers...")
            header_elements = page.locator(f"{results_table_selector} thead th")
            # Wait briefly for headers to potentially update after column toggles
            await page.wait_for_timeout(1000)
            headers = await header_elements.all_text_contents()
            clean_headers = [clean_header_key(h) for h in headers]
            logging.info(f"Current visible table headers (cleaned keys): {clean_headers}")

            # Find indices dynamically based on cleaned header keys
            header_map = {header: idx for idx, header in enumerate(clean_headers)}

            # --- Define desired columns and their expected keys ---
            desired_columns = {
                "track": "track",
                "artist": "artist",
                "key": "key",
                "bpm": "bpm",
                # Add any other specific columns you *always* want, using their cleaned key
                "duration": "duration",
                "camelot": "camelot",
                "acousticness": "acousticness",
                "danceability": "danceability",
                "energy": "energy",
                "instrumentalness": "instrumentalness",
                "liveness": "liveness",
                "loudness": "loudness",
                "speechiness": "speechiness",
                "valence": "valence",
                "popularity": "popularity",
                "release_date": "release_date",
                # Add links if needed later, e.g., "spotify": "spotify"
            }
            column_indices = {}
            missing_headers = []
            for name, key in desired_columns.items():
                if key in header_map:
                    column_indices[name] = header_map[key]
                else:
                    logging.warning(f"Could not find header for '{name}' (expected key: '{key}') in table headers.")
                    missing_headers.append(name)

            # Re-locate rows
            rows = await page.query_selector_all(f'{results_table_selector} tbody tr')
            logging.info(f"Re-located {len(rows)} rows in the table for final scrape.")
            playlist_data = []

            for i, row in enumerate(rows):
                columns = await row.query_selector_all('td')
                row_data = {}
                error_in_row = False

                # Extract data using dynamically found indices
                for name, index in column_indices.items():
                    if index < len(columns):
                        try:
                            cell_content = await columns[index].text_content()
                            row_data[name] = (cell_content or "").strip()
                        except Exception as e:
                            logging.error(f"Error reading cell for '{name}' (index {index}) in row {i+1}: {e}")
                            row_data[name] = "Error Parsing Cell"
                            error_in_row = True
                    else:
                        logging.warning(f"Row {i+1} is shorter than expected, missing column '{name}' at index {index}")
                        row_data[name] = "Missing Column"
                        error_in_row = True

                # Also try to get the track URL if 'track' column was found
                if "track" in column_indices and column_indices["track"] < len(columns):
                     try:
                          track_link_element = await columns[column_indices["track"]].query_selector('a')
                          track_url = await track_link_element.get_attribute('href') if track_link_element else None
                          if track_url and not track_url.startswith(('http:', 'https:')):
                               track_url = urllib.parse.urljoin(page.url, track_url)
                          row_data["track_url"] = track_url
                     except Exception as e:
                          logging.warning(f"Could not extract track_url from row {i+1}: {e}")
                          row_data["track_url"] = None

                if error_in_row:
                     row_data["parse_errors"] = True # Add a flag indicating issues

                playlist_data.append(row_data)

            results["data"] = playlist_data
            logging.info(f"Successfully processed {len(rows)} rows, generated {len(playlist_data)} entries.")
            if missing_headers:
                 logging.warning(f"Note: The following expected columns were not found in the table: {', '.join(missing_headers)}")

    except Exception as e:
        logging.error(f"An error occurred during Playwright operation: {e}", exc_info=True)
        results["error"] = f"An unexpected error occurred: {e}"

    return results

# --- Main Execution Block (Unchanged from previous) ---
async def main():
    """Constructs the direct URL and runs the scraping process."""
    print(f"--- Starting Direct Playlist Extraction for: {INPUT_PLAYLIST_URL} ---")
    print("Attempting to enable all columns via settings...")

    direct_url = create_songdata_search_url(INPUT_PLAYLIST_URL)
    extraction_result = await extract_songdata_playlist_direct(direct_url)

    print("\n--- Extraction Results ---")
    if extraction_result.get("error"):
        print(f"Error: {extraction_result['error']}")
    elif extraction_result.get("data") is not None:
        print(f"Playlist Data ({len(extraction_result['data'])} entries):")
        print(json.dumps(extraction_result["data"], indent=2))
        # Save the data in a json file
        with open('playlist_data.json', 'w') as f:
            json.dump(extraction_result["data"], f, indent=2)
        parse_errors = sum(1 for entry in extraction_result["data"] if "parse_errors" in entry or "Error Parsing Cell" in entry.values())
        if parse_errors > 0:
             print(f"\nNote: Encountered parsing errors in {parse_errors} row(s). Check logs for details.")

    else:
        print("Extraction finished, but no data or error was returned (unexpected state).")

    print("--- Extraction Finished ---")

if __name__ == '__main__':
    print("Please ensure Playwright browsers are installed: run 'playwright install chromium'")
    print("Using direct URL method. Will attempt to check all 'Columns' checkboxes.")
    print("!! IMPORTANT: Selectors for settings/columns button and dropdown were updated based on provided HTML. !!")
    asyncio.run(main())

