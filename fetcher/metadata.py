import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import json
import logging
import urllib.parse
import re # For cleaning header keys
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# --- Configuration ---
INPUT_PLAYLIST_URL = os.getenv("PLAYLIST_LINK") # Example: Spotify "lofi beats"

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