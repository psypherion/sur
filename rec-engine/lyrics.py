import re
import os
import json
import random
import logging
import asyncio
import requests
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
import sys # Import sys for command line arguments

# === Configuration ===
BASE_URL = "https://www.azlyrics.com/lyrics/"
# INPUT_JSON_PATH is a command-line argument
INPUT_JSON_PATH = None
OUTPUT_DIR = None # This will be derived from the input JSON path

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def sanitize_for_azlyrics_url(text: str) -> str:
    """
    Sanitizes text for use in AZLyrics URL paths.
    Based on observed patterns: lowercase, remove non-alphanumeric except underscore,
    remove common suffixes/prefixes/symbols, replace spaces with nothing.
    """
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r'(?i)^\s*the\s+', '', text) # Remove leading 'the ' case-insensitive
    text = re.sub(r'[^a-z0-9_]', '', text) # Keep only lowercase letters, numbers, and underscores
    text = re.sub(r'\b(remastered|remaster|remix|version|live|acoustic|feat|ft)\b', '', text) # Remove common words
    text = re.sub(r'\s+', '', text) # Remove any remaining whitespace (should be none after previous steps, but double-check)
    return text

def sanitize_for_filename(text: str) -> str:
    """
    Sanitizes text for use in filenames.
    Keeps spaces, removes invalid characters, and removes trailing/leading spaces/dots.
    """
    if not text:
        return "unknown"
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', text) # Replace invalid chars with underscore
    sanitized = re.sub(r'\s+', ' ', sanitized).strip() # Collapse multiple spaces and strip whitespace
    sanitized = re.sub(r'\.$', '', sanitized) # Remove trailing dot
    return sanitized or "unknown" # Return cleaned text or 'unknown' if empty


def construct_url(artist: str, song: str) -> str:
    """Constructs the AZLyrics URL from sanitized artist and song names."""
    sanitized_artist = sanitize_for_azlyrics_url(artist)
    sanitized_song = sanitize_for_azlyrics_url(song)
    if not sanitized_artist or not sanitized_song:
        logging.warning(f"Could not construct URL for Artist: '{artist}', Song: '{song}' due to sanitization failure.")
        return None # Return None if sanitization results in empty strings
    return f"{BASE_URL}{sanitized_artist}/{sanitized_song}.html"

# --- Modified function to load song data from JSON ---
def load_song_data(json_filepath: str) -> list:
    """
    Loads song data (full list of dictionaries) from a song_data.json file.

    Args:
        json_filepath: Path to the song_data.json file.

    Returns:
        A list of dictionaries loaded from the JSON data.
        Returns an empty list if file not found or parsing fails.
    """
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                 logging.info(f"Successfully loaded {len(data)} entries from '{json_filepath}'.")
                 return data
            else:
                 logging.error(f"JSON data in '{json_filepath}' is not a list.")
                 return [] # Return empty list if data is not a list

    except FileNotFoundError:
        logging.error(f"Error: Input JSON file not found at '{json_filepath}'")
        return [] # Return empty list on file not found
    except json.JSONDecodeError:
        logging.error(f"Error: Could not parse JSON file at '{json_filepath}'")
        return [] # Return empty list on JSON parsing error
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading '{json_filepath}': {e}", exc_info=True)
        return [] # Return empty list on other errors

# --- Fallback Fetch (using requests) ---
def fallback_fetch(url: str) -> str:
    """
    Attempts to fetch lyrics using requests as a fallback.
    """
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    logging.debug(f"Attempting fallback fetch for URL: {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # AZLyrics embeds lyrics within and <script> tags
        match = re.search(
            r"(.*?)<script[^>]*>",
            resp.text, re.DOTALL | re.IGNORECASE
        )
        if match:
            lyrics = match.group(1).strip()
            # Optional: Further clean the extracted lyrics if needed (e.g., remove HTML breaks <br>)
            # lyrics = lyrics.replace('<br>', '\n').replace('<br/>', '\n') # Example
            logging.debug("Lyrics content pattern matched in fallback.")
            return lyrics
        else:
            logging.debug("Lyrics content pattern not found in fallback response.")
            return None
    except requests.exceptions.RequestException as e:
        logging.debug(f"Requests fallback failed for {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during fallback fetch for {url}: {e}", exc_info=True)
        return None


# --- Playwright Fetch ---
async def fetch_lyrics_playwright(page, url: str) -> str:
    """
    Attempts to fetch lyrics using Playwright with retry logic.
    """
    logging.debug(f"Attempting Playwright fetch for URL: {url}")
    for attempt in range(3): # Retry up to 3 times
        try:
            # Set random user agent and viewport before navigation attempt
            await page.set_extra_http_headers({"User-Agent": random.choice(USER_AGENTS)})
            await page.set_viewport_size({
                "width": random.choice([1280, 1366, 1440, 1920]),
                "height": random.choice([720, 768, 900, 1080])
            })
            # Add init script for stealth (only needs to be added once per context, but safe to call)
            # await context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => false});") # Stealth handles this

            # Introduce random delay before navigation
            await asyncio.sleep(random.uniform(2, 5))

            # Navigate to the URL
            await page.goto(url, timeout=20000, wait_until='domcontentloaded') # Use domcontentloaded, often faster

            # Check for common AZLyrics "Not Found" page indicators if needed
            # not_found_text = await page.locator("body:has-text('404 - NOT FOUND')").count()
            # if not_found_text > 0:
            #      logging.warning(f"AZLyrics page not found for {url}")
            #      return None # Indicate page not found

            html = await page.content()

            # AZLyrics embeds lyrics within and <script> tags
            match = re.search(
                r"<!--\s*Usage of azlyrics\.com content.*?-->(.*?)<script>",
                html, re.DOTALL | re.IGNORECASE
            )
            if match:
                lyrics = match.group(1).strip()
                # Optional: Further clean the extracted lyrics if needed (e.g., remove HTML breaks <br>)
                lyrics = lyrics.replace('<br>', '\n').replace('<br/>', '\n') # Example
                logging.debug("Lyrics content pattern matched via Playwright.")
                return lyrics
            else:
                logging.debug("Lyrics content pattern not found on the page via Playwright.")
                return None # Pattern not found
        except PlaywrightTimeoutError:
            logging.warning(f"[Retry {attempt+1}/3] Playwright navigation or wait timed out for {url}.")
            await asyncio.sleep(random.uniform(5, 10)) # Longer wait on timeout
        except Exception as e:
            logging.warning(f"[Retry {attempt+1}/3] Playwright failed for {url}: {e}")
            await asyncio.sleep(random.uniform(3, 6)) # Wait before retry
    logging.error(f"Playwright failed to fetch lyrics after {attempt+1} attempts for {url}")
    return None # Failed after retries


async def main():

    # global INPUT_JSON_PATH # Declare global to assign
    INPUT_JSON_PATH = "/home/psyph3ri0n/Documents/projects-2025/sur/downloads/spoti-test/song_details.json"

    if not os.path.exists(INPUT_JSON_PATH):
        logging.error(f"Input file not found: {INPUT_JSON_PATH}")
        sys.exit(1)

    # Derive output directory based on input JSON file location
    json_dir = os.path.dirname(INPUT_JSON_PATH)
    global OUTPUT_DIR # Declare global to assign
    OUTPUT_DIR = os.path.join(json_dir, "lyrics")  

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Output directory set to: {OUTPUT_DIR}")

    # Load song data from the JSON file (get the full list of dictionaries)
    songs_data_list = load_song_data(INPUT_JSON_PATH)

    if not songs_data_list:
        logging.error("No song data loaded from JSON or JSON is empty/invalid. Exiting.")
        sys.exit(1)

    async with async_playwright() as p:
        # Launch browser with stealth plugin
        browser = await p.chromium.launch(headless=True) # Set headless=False for debugging if needed
        context = await browser.new_context()
        await stealth_async(context) # Apply stealth to the context

        page = await context.new_page()

        processed_count = 0
        found_lyrics_count = 0

        # Iterate directly through the dictionaries in the loaded list
        for item in songs_data_list:
            processed_count += 1
            # Safely get song and artist names from the dictionary
            song = item.get('spotify_song_name', '')
            artist = item.get('spotify_artist_name', '')

            if not song or not artist:
                 logging.warning(f"[{processed_count}/{len(songs_data_list)}] Skipping entry {item.get('id', 'N/A')} due to missing song or artist name in JSON.")
                 continue # Skip if essential info is missing


            logging.info(f"[{processed_count}/{len(songs_data_list)}] Processing: {artist} - {song}")

            # Construct the AZLyrics URL
            url = construct_url(artist, song)

            if not url:
                logging.error(f"[{processed_count}/{len(songs_data_list)}] ✘ Skipping '{artist} - {song}' due to invalid URL construction.")
                # Optionally add a status to the item dictionary, e.g., item['lyrics_status'] = 'url_failed'
                continue # Skip to the next song

            logging.info(f"[{processed_count}/{len(songs_data_list)}] Fetching URL: {url}")

            # Playwright fetch attempt
            lyrics = await fetch_lyrics_playwright(page, url)

            if not lyrics:
                logging.warning(f"[{processed_count}/{len(songs_data_list)}] Playwright failed for {artist} - {song}. Trying fallback...")
                # Fallback fetch attempt
                lyrics = fallback_fetch(url)

            if lyrics:
                # Use original song and artist names (sanitized for filename)
                filename = f"{sanitize_for_filename(artist)} - {sanitize_for_filename(song)}.txt"
                file_path = os.path.join(OUTPUT_DIR, filename)
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(lyrics.strip()) # Write cleaned lyrics

                    # --- Add/Update the lyrics_file_path in the current item dictionary ---
                    item['lyrics_file_path'] = file_path

                    logging.info(f"[{processed_count}/{len(songs_data_list)}] ✔ Saved lyrics to: {file_path} and updated JSON data.")
                    found_lyrics_count += 1
                except IOError as e:
                     logging.error(f"[{processed_count}/{len(songs_data_list)}] Error saving file {file_path}: {e}")
                     # Optionally add a status to the item dictionary, e.g., item['lyrics_status'] = 'save_failed'
                except Exception as e:
                     logging.error(f"[{processed_count}/{len(songs_data_list)}] An unexpected error occurred saving file {file_path}: {e}", exc_info=True)
                     # Optionally add a status to the item dictionary
            else:
                logging.error(f"[{processed_count}/{len(songs_data_list)}] ✘ Lyrics not found for {artist} - {song} after all attempts.")
                # Optionally add a status to the item dictionary, e.g., item['lyrics_status'] = 'not_found'


        await browser.close()
        logging.info("Browser closed.")

        # --- Save the modified song data back to the JSON file ---
        try:
            with open(INPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(songs_data_list, f, indent=2, ensure_ascii=False)
            logging.info(f"\nSuccessfully saved updated song data (including lyrics paths) to '{INPUT_JSON_PATH}'")
        except IOError as e:
             logging.error(f"Error saving updated JSON file '{INPUT_JSON_PATH}': {e}")
        except Exception as e:
             logging.error(f"An unexpected error occurred saving updated JSON file '{INPUT_JSON_PATH}': {e}", exc_info=True)


        print(f"\n--- Script Finished ---")
        print(f"Processed {processed_count} songs from JSON.")
        print(f"Successfully found and saved lyrics for {found_lyrics_count} songs.")
        print(f"Lyrics saved to: {OUTPUT_DIR}")
        print(f"Updated song data saved to: {INPUT_JSON_PATH}")


if __name__ == "__main__":
    asyncio.run(main())