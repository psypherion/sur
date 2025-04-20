import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import json
import logging
import urllib.parse
import re # For cleaning header keys
import os # Import os for file path operations
import sys # Import sys for command line arguments

# --- Configuration ---
# Base directory where downloads and song_details.json are saved
DOWNLOADS_BASE_DIR = "downloads"
# Placeholder for the playlist URL - will be taken from command line
INPUT_PLAYLIST_URL = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions (from your previous script) ---
def get_playlist_id_from_url(playlist_url: str) -> str:
    """Extracts playlist ID from Spotify URL."""
    try:
        id_si = playlist_url.split('playlist/')
        if len(id_si) < 2:
             raise ValueError("Invalid playlist link format")
        playlist_part = id_si[1]
        if '?si' in playlist_part:
            return playlist_part.split('?')[0]
        else:
            return playlist_part
    except ValueError as e:
        logging.error(f"Error parsing playlist ID from link: {e}")
        sys.exit(1)

def sanitize_foldername(name):
    """Sanitizes string for use as a folder name."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized or "Unknown_Playlist"


# --- Function to create the direct songdata.io search URL ---
def create_songdata_search_url(playlist_url: str) -> str:
    """
    Creates the direct search URL for songdata.io by URL-encoding the playlist link.
    """
    base_url = "https://songdata.io/search?query="
    encoded_playlist_url = urllib.parse.quote(playlist_url, safe='')
    direct_url = base_url + encoded_playlist_url
    logging.info(f"Constructed songdata.io search URL: {direct_url}")
    return direct_url

# --- Function to clean header text for JSON keys ---
def clean_header_key(header_text):
    """Cleans header text to be a valid and readable JSON key."""
    if not header_text:
        return "unknown_header"
    # Replace '#' with 'rank', remove special chars, lowercase, replace spaces with underscores
    key = header_text.strip().lower()
    key = key.replace('#', 'rank')
    # Keep alphanumeric, space, hyphen, underscore. Remove others.
    key = re.sub(r'[^\w\s-]', '', key)
    key = re.sub(r'[-\s]+', '_', key) # Replace spaces/hyphens with underscores
    return key or "unknown_header" # Return cleanup result or default

# --- Playwright Scraping Function (Updated for Settings Interaction) ---
async def extract_songdata_playlist_direct(direct_search_url: str):
    """
    Navigates directly to the songdata.io search results page,
    opens the settings dropdown, checks all checkboxes,
    and scrapes the resulting tracklist table.
    """
    logging.info(f"Starting songdata.io data extraction from: {direct_search_url}")
    results = {"data": [], "error": None} # Initialize data as empty list
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
            # Increased timeout and added 'domcontentloaded' wait state as it's often faster
            await page.goto(direct_search_url, timeout=120000, wait_until='domcontentloaded')

            # --- Wait for Initial Table (or alternative indicators) ---
            results_table_selector = "#table_chart" # Confirmed from HTML
            no_results_selector = "div.col-md-12:has(h3:text('Could not find playlist'))" # Selector for "Could not find playlist" message
            rate_limit_selector = "body:has-text('rate limit')" # Basic check for rate limit text

            logging.info(f"Waiting for results table ('{results_table_selector}') or no results/rate limit indicators...")
            try:
                # Wait for one of the selectors to appear
                await page.wait_for_selector(f"{results_table_selector}, {no_results_selector}, {rate_limit_selector}", state="visible", timeout=180000)

                # Check which selector appeared first
                is_table_visible = await page.locator(results_table_selector).first.is_visible()
                is_no_results_visible = await page.locator(no_results_selector).first.is_visible()
                is_rate_limit_visible = await page.locator(rate_limit_selector).first.is_visible()


                if is_rate_limit_visible:
                     logging.error("Rate limit likely hit on songdata.io.")
                     results["error"] = "Rate limit likely hit on songdata.io."
                     if browser: await browser.close()
                     return results
                elif is_no_results_visible:
                     logging.warning("Songdata.io reported: Could not find playlist.")
                     results["error"] = "Songdata.io reported: Could not find playlist."
                     if browser: await browser.close()
                     return results
                elif is_table_visible:
                    logging.info("Initial results table found.")
                else: # Should not happen if one of the above was visible, but as a fallback
                     logging.error("Timeout waiting for results table or error messages.")
                     results["error"] = "Timeout waiting for page content. Playlist might be invalid/private, or site issues."
                     if browser: await browser.close()
                     return results


            except PlaywrightTimeoutError:
                 logging.error(f"Timeout waiting for initial results table or error messages on direct URL.")
                 results["error"] = "Timeout waiting for page content. Playlist might be invalid/private, or site issues."
                 if browser: await browser.close()
                 return results


            # --- Interact with Settings/Columns Dropdown ---
            settings_button_selector = "button.settingbtn" # Confirmed from new HTML
            dropdown_menu_selector = "div#setting.setting-dropdown" # Confirmed from new HTML

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

                # Select all checkboxes that toggle columns within the dropdown
                checkbox_locators = dropdown_menu.locator("input[type='checkbox'].toggle-col")
                count = await checkbox_locators.count()
                logging.info(f"Found {count} toggle checkboxes in the settings dropdown.")

                checked_count = 0
                # Iterate through checkboxes and check if not already checked
                for i in range(count):
                    checkbox = checkbox_locators.nth(i)
                    try:
                        # Check if the checkbox is already checked
                        is_checked = await checkbox.is_checked()
                        checkbox_id = await checkbox.get_attribute('id')
                        label_locator = dropdown_menu.locator(f"label[for='{checkbox_id}']")
                        label_text = await label_locator.text_content() if await label_locator.count() > 0 else f"Label for {checkbox_id}"

                        logging.debug(f"Checkbox {i+1} ('{label_text.strip()}'): Already checked = {is_checked}")

                        if not is_checked:
                            logging.info(f"Checking checkbox {i+1}: '{label_text.strip()}'")
                            # Using click on the label is often more robust for custom checkboxes
                            await label_locator.click(timeout=5000)
                            # Alternative: await checkbox.check(force=True, timeout=5000) # force=True might help if label click fails
                            checked_count += 1
                            await page.wait_for_timeout(50) # Small pause after checking each box
                    except PlaywrightTimeoutError:
                         logging.warning(f"Timeout interacting with checkbox {i+1}. Skipping.")
                    except Exception as e:
                         logging.warning(f"Error interacting with checkbox {i+1}: {e}. Skipping.", exc_info=True)


                logging.info(f"Finished attempting to check boxes. {checked_count} boxes were newly checked.")

                # Click the settings button again to attempt to close the dropdown
                logging.info("Clicking settings button again to close dropdown...")
                # Check if the settings button is still visible before clicking again
                if await settings_button.is_visible():
                    await settings_button.click()
                    await page.wait_for_timeout(500) # Wait a moment for UI updates
                else:
                     logging.warning("Settings button not visible after checking boxes, cannot explicitly close dropdown.")


            except PlaywrightTimeoutError as e:
                logging.warning(f"Could not find or interact with the Settings dropdown within timeout: {e}")
                logging.warning("Proceeding to scrape table with currently visible columns.")
            except Exception as e:
                 logging.error(f"An unexpected error occurred during settings interaction: {e}", exc_info=True)
                 logging.warning("Attempting to continue scraping despite settings error.")


            # --- Extract Data from Table ---
            # Re-fetch headers AFTER potential column changes
            logging.info("Fetching final table headers...")
            # Wait briefly for headers to potentially update after column toggles
            await page.wait_for_timeout(1000) # Give the table a moment to re-render columns
            header_elements = page.locator(f"{results_table_selector} thead th")

            headers = await header_elements.all_text_contents()
            if not headers:
                logging.warning("No headers found in the table after trying to enable all columns. Table might be empty or structure changed.")
                # Continue without scraping rows, data will be empty list
                return results # Return empty data list

            clean_headers = [clean_header_key(h) for h in headers]
            logging.info(f"Current visible table headers (cleaned keys): {clean_headers}")

            # Find indices dynamically based on cleaned header keys
            header_map = {header: idx for idx, header in enumerate(clean_headers)}

            # Define desired columns and their expected keys
            # Use a dictionary to map a standard name to the cleaned key
            # This makes merging easier later
            desired_columns = {
                "songdata_track": "track", # Renamed to avoid conflict with 'track_url' or future keys
                "songdata_artist": "artist", # Renamed
                "key": "key",
                "bpm": "bpm",
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
                "genres": "genres", # Added genres as per songdata.io data
                "label": "label", # Added label
                # Add links if needed later, e.g., "spotify_link": "spotify"
            }
            column_indices = {}
            missing_headers = []
            for name, key in desired_columns.items():
                if key in header_map:
                    column_indices[name] = header_map[key]
                else:
                    # Log missing headers, but continue if core columns are present
                    if key in ["track", "artist"]: # Check if essential columns are missing
                         logging.error(f"CRITICAL: Could not find essential header for '{name}' (expected key: '{key}') in table headers. Cannot proceed with reliable scraping.")
                         results["error"] = f"Missing essential table column: '{key}'"
                         if browser: await browser.close()
                         return results
                    logging.warning(f"Could not find header for '{name}' (expected key: '{key}') in table headers.")
                    missing_headers.append(name)

            # Find table body rows
            rows = await page.locator(f'{results_table_selector} tbody tr').all() # Use locator.all()
            logging.info(f"Found {len(rows)} rows in the table for scraping.")
            songdata_data = [] # This list will hold the scraped songdata.io dictionaries

            for i, row in enumerate(rows):
                # Use locator.all() on row as well
                columns = await row.locator('td').all()
                row_data = {}
                error_in_row = False

                # Extract data using dynamically found indices
                for name, index in column_indices.items():
                    if index < len(columns):
                        try:
                            cell_content = await columns[index].text_content()
                            # Clean up the cell content (e.g., remove trailing/leading whitespace)
                            cleaned_content = (cell_content or "").strip()

                            # --- Optional: Type Conversion/Cleaning for specific columns ---
                            if name in ["bpm", "acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "valence", "popularity"]:
                                try:
                                    # Attempt to convert to float, handle potential errors (e.g., N/A, '-')
                                    row_data[name] = float(cleaned_content) if cleaned_content and cleaned_content != '-' and cleaned_content.lower() != 'n/a' else None
                                except ValueError:
                                     logging.warning(f"Could not convert '{name}' value '{cleaned_content}' to float in row {i+1}.")
                                     row_data[name] = cleaned_content # Keep as string if conversion fails
                            # Add more specific cleaning if needed (e.g., duration format)
                            elif name == "duration":
                                # Basic attempt to keep numerical part if it contains time format like mm:ss
                                # A more robust parser might be needed for complex formats
                                duration_match = re.search(r'(\d+)', cleaned_content)
                                row_data[name] = duration_match.group(1) if duration_match else cleaned_content
                            else:
                                row_data[name] = cleaned_content # Keep as string

                        except Exception as e:
                            logging.error(f"Error reading cell for '{name}' (index {index}) in row {i+1}: {e}")
                            row_data[name] = "Error Parsing Cell"
                            error_in_row = True
                    else:
                        logging.warning(f"Row {i+1} is shorter than expected ({len(columns)} columns), missing column '{name}' at index {index}")
                        row_data[name] = "Missing Column"
                        error_in_row = True

                # Also try to get the track URL if 'songdata_track' column was found
                if "songdata_track" in column_indices and column_indices["songdata_track"] < len(columns):
                     try:
                          # Assuming the link is within the cell corresponding to 'songdata_track'
                          track_link_element = await columns[column_indices["songdata_track"]].locator('a').first
                          track_url = await track_link_element.get_attribute('href') if track_link_element else None
                          # Ensure the URL is absolute if it's relative
                          if track_url and not track_url.startswith(('http://', 'https://')):
                               track_url = urllib.parse.urljoin(page.url, track_url)
                          row_data["songdata_track_url"] = track_url # Renamed key
                     except Exception as e:
                          # Log warning but don't necessarily mark row as error unless critical
                          logging.warning(f"Could not extract songdata_track_url from row {i+1}: {e}")
                          row_data["songdata_track_url"] = None

                if error_in_row:
                     row_data["parse_errors"] = True # Add a flag indicating issues

                songdata_data.append(row_data) # Add the scraped row data to our list

            results["data"] = songdata_data # Assign the scraped data to the results dictionary
            logging.info(f"Successfully scraped data for {len(songdata_data)} songs from songdata.io.")
            if missing_headers:
                 logging.warning(f"Note: The following expected columns were not found in the table: {', '.join(missing_headers)}")

    except Exception as e:
        logging.error(f"An error occurred during Playwright operation: {e}", exc_info=True)
        results["error"] = f"An unexpected error occurred during scraping: {e}"

    finally:
        if browser:
            await browser.close()
            logging.info("Browser closed.")

    return results

# --- Merging Function ---
def merge_song_data(spotify_youtube_data, songdata_data):
    """
    Merges songdata.io data into the Spotify/YouTube data based on song and artist name.

    Args:
        spotify_youtube_data: List of dictionaries from song_details.json.
        songdata_data: List of dictionaries scraped from songdata.io.

    Returns:
        A list of dictionaries with merged data.
    """
    logging.info("Starting data merging process.")
    merged_results = []

    # Create a lookup dictionary for songdata.io data for efficient matching
    songdata_lookup = {}
    for item in songdata_data:
        # Use lower case for robust matching, handle potential missing keys
        track = item.get('songdata_track', '').lower()
        artist = item.get('songdata_artist', '').lower()
        if track and artist: # Only add to lookup if both track and artist are present
            # Use a tuple as the key
            key = (track, artist)
            # Handle potential duplicates in songdata.io results by keeping the first one found
            if key not in songdata_lookup:
                songdata_lookup[key] = item
            else:
                 logging.warning(f"Duplicate songdata.io entry found for '{track}' by '{artist}'. Keeping the first one.")
        else:
             logging.warning(f"Skipping songdata.io entry due to missing track or artist: {item}")


    # Iterate through the Spotify/YouTube data and merge
    for spotify_item in spotify_youtube_data:
        # Create lookup key from Spotify data, using lower case
        spotify_track = spotify_item.get('spotify_song_name', '').lower()
        spotify_artist = spotify_item.get('spotify_artist_name', '').lower()
        lookup_key = (spotify_track, spotify_artist)

        # Find matching songdata.io item
        matching_songdata_item = songdata_lookup.get(lookup_key)

        # Create the merged dictionary
        merged_item = spotify_item.copy() # Start with a copy of the Spotify/YouTube data

        if matching_songdata_item:
            logging.debug(f"Match found for '{spotify_track}' by '{spotify_artist}'. Merging songdata info.")
            # Add the songdata.io details under a nested key
            # Exclude the songdata_track and songdata_artist keys from the nested dict
            # as they are redundant with the top-level spotify names after matching
            songdata_info_to_nest = {k: v for k, v in matching_songdata_item.items() if k not in ['songdata_track', 'songdata_artist']}
            merged_item["songdata_info"] = songdata_info_to_nest
        else:
            logging.warning(f"No songdata.io match found for '{spotify_track}' by '{spotify_artist}'.")
            # Add an empty songdata_info dictionary to maintain consistent structure
            merged_item["songdata_info"] = {}

        merged_results.append(merged_item)

    logging.info(f"Merging complete. Generated {len(merged_results)} merged entries.")
    return merged_results


# --- Main Orchestration Block ---
async def main():
    """
    Orchestrates the process: loads existing data, scrapes songdata.io,
    merges the data, and saves the final result.
    """
    # Get playlist URL from command line arguments
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <spotify_playlist_url>")
        sys.exit(1)

    global INPUT_PLAYLIST_URL # Declare global to assign
    INPUT_PLAYLIST_URL = sys.argv[1]

    print(f"--- Processing Playlist: {INPUT_PLAYLIST_URL} ---")

    # 1. Determine the playlist directory path
    playlist_id = get_playlist_id_from_url(INPUT_PLAYLIST_URL)
    # Note: The Spotify downloader script gets the actual playlist name from the API
    # and uses that for the folder. We need to replicate that here or get the name.
    # A simpler assumption for this script is that the folder name is a sanitized
    # version of the playlist ID or a default name if the playlist name wasn't available.
    # Let's assume the Spotify downloader saved a 'playlist_name.txt' or similar
    # or we can re-fetch the name (requires Spotify API token here too).
    # For simplicity, let's assume the Spotify downloader uses a sanitized playlist name
    # which we cannot reliably know without the API call.
    # A more robust approach would be to pass the actual directory path from the first script.
    # For now, let's make an educated guess or require a convention.
    # Let's try to load the original song_details.json first to get the name if it exists.

    spotify_youtube_data = None
    playlist_dir_path = None # Initialize to None

    # Attempt to find the song_details.json file
    # A common approach is to search directories under DOWNLOADS_BASE_DIR
    logging.info(f"Searching for song_details.json in subdirectories of '{DOWNLOADS_BASE_DIR}'...")
    song_details_filename = "song_details.json"
    found_song_details_path = None

    for dirpath, dirnames, filenames in os.walk(DOWNLOADS_BASE_DIR):
        if song_details_filename in filenames:
            # Found it, now try to match if this file belongs to the input playlist URL
            # We can potentially load the JSON and check one of the entries,
            # or try to infer the playlist name from the folder name.
            # Inferring from folder name is risky as sanitization can vary.
            # Loading and checking is more reliable but requires loading potentially large files.
            # Let's assume the folder name is predictable (e.g., sanitized playlist name from Spotify API).
            # This requires the Spotify downloader to print the folder name or follow a strict convention.
            # A strict convention: folder name is sanitized(playlist ID).
            # Let's fall back to iterating and checking the *contents* if a strict convention isn't used.

            potential_json_path = os.path.join(dirpath, song_details_filename)
            try:
                with open(potential_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data and isinstance(data, list) and len(data) > 0:
                        # Check if any entry contains the playlist URL or ID or derived name
                        # This is still complex without a guaranteed link back in song_details.json
                        # Let's assume the folder name is the sanitized playlist name.
                        # We need the playlist name first... which the first script gets from the API.

                        # *Alternative Approach (Simpler for this script):*
                        # Require the user to provide the *directory path* as a second argument.
                        # This is much more reliable.

                        logging.warning(f"Found potential song_details.json at '{potential_json_path}', but matching it to the input URL without playlist name is difficult.")
                        # Let's try to load the playlist name from the .txt file if it exists
                        playlist_name_txt_path = os.path.join(dirpath, os.path.basename(dirpath) + ".txt")
                        if os.path.exists(playlist_name_txt_path):
                            try:
                                with open(playlist_name_txt_path, 'r', encoding='utf-8') as txt_f:
                                     first_line = txt_f.readline()
                                     if first_line.startswith("Playlist Name:"):
                                         inferred_playlist_name = first_line.replace("Playlist Name:", "").strip()
                                         inferred_dir_name = sanitize_foldername(inferred_playlist_name)
                                         if os.path.basename(dirpath) == inferred_dir_name:
                                             logging.info(f"Inferred playlist '{inferred_playlist_name}' matches folder '{os.path.basename(dirpath)}'. Using this path.")
                                             found_song_details_path = potential_json_path
                                             playlist_dir_path = dirpath
                                             spotify_youtube_data = data
                                             break # Found and matched, exit search
                            except Exception as txt_e:
                                logging.warning(f"Could not read playlist name from text file at '{playlist_name_txt_path}': {txt_e}")
                                pass # Continue searching directories

            except Exception as e:
                logging.warning(f"Could not load or parse JSON file at '{potential_json_path}': {e}")
                pass # Continue searching directories

    if not found_song_details_path or not spotify_youtube_data:
        logging.error(f"Could not find or load song_details.json for playlist URL: {INPUT_PLAYLIST_URL}")
        logging.info(f"Expected the file in a subdirectory of '{DOWNLOADS_BASE_DIR}' named after the playlist.")
        logging.info("Please ensure your first script ran successfully and created 'song_details.json' in the correct folder.")
        sys.exit(1)
    else:
         logging.info(f"Successfully loaded existing song data from '{found_song_details_path}'.")
         logging.info(f"Inferred playlist directory: '{playlist_dir_path}'")


    # 2. Scrape data from songdata.io
    print("\n--- Scraping songdata.io ---")
    direct_url = create_songdata_search_url(INPUT_PLAYLIST_URL)
    extraction_result = await extract_songdata_playlist_direct(direct_url)

    songdata_data = extraction_result.get("data")
    scrape_error = extraction_result.get("error")

    if scrape_error:
        print(f"\n--- Songdata.io Scraping Error ---")
        print(f"Error: {scrape_error}")
        if songdata_data is not None:
            print(f"Note: Partial data ({len(songdata_data)} entries) might have been scraped before the error.")
            # Continue to merge with partial data
        else:
             print("No data was scraped from songdata.io.")
             # Cannot merge if no songdata_data
             print("--- Merging Skipped Due to Scraping Error ---")
             sys.exit(1) # Exit if scraping failed completely

    elif not songdata_data:
         print("\n--- Songdata.io Scraping Complete ---")
         print("No song data found on songdata.io for this playlist.")
         # Continue to merge, the songdata_info will be empty for all songs
         songdata_data = [] # Ensure it's an empty list for merging

    else:
        print("\n--- Songdata.io Scraping Complete ---")
        print(f"Scraped data for {len(songdata_data)} songs.")
        parse_errors = sum(1 for entry in songdata_data if "parse_errors" in entry or "Error Parsing Cell" in entry.values())
        if parse_errors > 0:
             print(f"Note: Encountered parsing errors in {parse_errors} scraped rows. Check logs for details.")


    # 3. Merge the data
    print("\n--- Merging Data ---")
    merged_data = merge_song_data(spotify_youtube_data, songdata_data)


    # 4. Save the final merged data
    song_data_json_path = os.path.join(playlist_dir_path, "song_data.json")
    try:
        with open(song_data_json_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully saved merged song data to '{song_data_json_path}'")
        print(f"Final output contains {len(merged_data)} entries.")

    except Exception as e:
        logging.error(f"Error saving merged data to JSON: {e}", exc_info=True)
        print(f"Error saving merged data: {e}")

    print("\n--- Process Finished ---")


if __name__ == '__main__':
    print("Please ensure Playwright browsers are installed: run 'playwright install chromium'")
    print("This script requires 'song_details.json' generated by the first script to exist.")
    print("Usage: python your_playwright_script_name.py <spotify_playlist_url>")

    asyncio.run(main())