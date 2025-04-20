import re
import os
import json
import random
import logging
import asyncio
import requests
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async

# Configs
BASE_URL = "https://www.azlyrics.com/lyrics/"
INPUT_FILE = "fetcher/downloads/Rock playlist/Rock playlist.txt"
PROXIES_FILE = "working_proxies.txt"
OUTPUT_DIR = "fetcher/downloads/Rock playlist/lyrics"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

import requests
import json

def validate_proxies(proxy_file, output_file="working_proxies.txt"):
    with open(proxy_file, "r") as f:
        raw = json.load(f)

    proxies = [p["proxy"] for p in raw.get("proxies", [])]
    good = []

    for proxy in proxies:
        try:
            resp = requests.get("https://httpbin.org/ip", proxies={"http": proxy, "https": proxy}, timeout=5)
            if resp.status_code == 200:
                print(f"[✓] {proxy}")
                good.append({"proxy": proxy, "alive": True})
            else:
                print(f"[x] {proxy} failed with status: {resp.status_code}")
        except Exception as e:
            print(f"[x] {proxy} died: {e}")

    with open(output_file, "w") as out:
        json.dump({"proxies": good}, out, indent=2)
    print(f"\nSaved {len(good)} working proxies to {output_file}")


def sanitize(text: str) -> str:
    text = re.sub(r'(?i)^\s*the\s+', '', text)  # remove 'the'
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    text = re.sub(r'\s+', '', text)
    return text.strip()


def construct_url(artist: str, song: str) -> str:
    return f"{BASE_URL}{sanitize(artist.lower())}/{sanitize(song.lower())}.html"


def parse_song_artist(filepath: str) -> list:
    songs, song, artist = [], None, None
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
    # Validate proxies : 
    validate_proxies("proxies.txt")
    # Load working proxies
    with open(PROXIES_FILE, 'r') as f:
        proxy_data = json.load(f)
    proxies = [p["proxy"] for p in proxy_data.get("proxies", []) if p.get("alive")]

    songs = parse_song_artist(INPUT_FILE)
    async with async_playwright() as p:
        for song, artist in songs:
            url = construct_url(artist, song)
            proxy = random.choice(proxies)
            logging.info(f"Using proxy {proxy} for {artist} - {song}")
            try:
                browser = await p.chromium.launch(headless=True, proxy={"server": proxy})
                page = await browser.new_page()
                await stealth_async(page)

                await page.set_extra_http_headers({
                    "User-Agent": random.choice(USER_AGENTS),
                    "Accept-Language": "en-US,en;q=0.9",
                })
                await page.set_viewport_size({
                    "width": random.choice([1280, 1366, 1440, 1920]),
                    "height": random.choice([720, 768, 900, 1080])
                })
                await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => false});")

                await asyncio.sleep(random.uniform(5, 15))  # Random delay

                lyrics = await fetch_lyrics_playwright(page, url)
                await browser.close()

                if not lyrics:
                    logging.warning(f"Trying fallback for {artist} - {song}")
                    lyrics = fallback_fetch(url)

                if lyrics:
                    filename = f"{OUTPUT_DIR}/{sanitize(artist)} - {sanitize(song)}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(lyrics)
                    logging.info(f"Saved: {filename}")
                else:
                    logging.error(f"Lyrics not found for {artist} - {song}")

            except Exception as e:
                logging.error(f"Proxy/browser fail for {url} - {e}")


if __name__ == "__main__":
    asyncio.run(main())
