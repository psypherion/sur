import os
import re
import json
import csv
import pandas as pd
from fuzzywuzzy import process, fuzz
from unidecode import unidecode

# --- Configuration ---
PLAYLIST_JSON_PATH = "playlist_data.json"  # Assuming playlist JSON is in the same dir
# Direct list of your actual MP3 file paths
ACTUAL_SONG_PATHS = [
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Kinks - All Day And All Of The Night (Official Audio).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Rolling Stones - Gimme Shelter (Official Lyric Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Nirvana - Something In The Way (Audio).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/505.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Aerosmith - Dream On (Audio).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Nirvana - Smells Like Teen Spirit (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Unforgiven (Remastered).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Nickelback - How You Remind Me [OFFICIAL VIDEO].mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Green Day - Brain Stew_Jaded [Official Music Video].mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Gorillaz - Feel Good Inc. (Official Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Nirvana - Lithium (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Foo Fighters - All My Life (Official HD Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/You Really Got Me (2014 Remaster).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Blur - Song 2 (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Who - Baba O\'Riley (Lyric Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Def Leppard - Hysteria (Official Lyric Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The White Stripes - Seven Nation Army (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Iron Maiden-The Trooper (2015 Remaster).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Green Day - Holiday [Official Music Video] [4K Upgrade].mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/KALEO - Way Down We Go (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Armageddon • I Don\'t Want to Miss a Thing • Aerosmith.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Pour Some Sugar On Me (Remastered 2017).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Animals - House Of The Rising Sun (Music Video) [4K HD].mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Who - Who Are You (Promo Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Def Leppard - Photograph.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Toto - Africa (Official HD Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Rolling Stones - Beast of Burden (Official Lyric Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Rolling Stones - You Can’t Always Get What You Want (Official Lyric Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Cage The Elephant - Ain\'t No Rest For The Wicked (Official Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Police - Message In A Bottle (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Killers - Mr. Brightside (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Finger Eleven - Paralyzer (Official HD Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/a-ha - Take On Me (Official Video) [Remastered in 4K].mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Nirvana - In Bloom (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Led Zeppelin - Houses of the Holy (Remaster) [Official Full Album].mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Journey - Separate Ways (Worlds Apart) (Official HD Video - 1983).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Pearl Jam - Alive (Official Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Iron Man (2012 Remaster).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Beastie Boys - (You Gotta) Fight For Your Right (To Party) (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Foster The People - Pumped Up Kicks (Official Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Foo Fighters - The Sky Is A Neighborhood (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Black Sabbath ~ War Pigs.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Chain (2004 Remaster).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Rolling Stones - Sympathy For The Devil (Official Lyric Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Master of Puppets (Remastered).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Pearl Jam - Even Flow (Official Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Police - Don\'t Stand So Close To Me (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Bon Jovi - It\'s My Life (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/U2 - I Still Haven\'t Found What I\'m Looking For (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Nirvana - Heart-Shaped Box.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Rolling Stones - Paint It, Black (Official Lyric Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Aerosmith - Sweet Emotion.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Beastie Boys - Intergalactic (2009 Digital Remaster).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Ram Jam - Black Betty.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Police - Roxanne (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Foo Fighters - My Hero (Official HD Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Ozzy Osbourne - Crazy Train (Official Animated Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Red Hot Chili Peppers - Californication (Official Music Video) [HD UPGRADE].mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Tom Petty And The Heartbreakers - Mary Jane\'s Last Dance (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Aerosmith - Walk This Way (Audio).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Bon Jovi - Wanted Dead Or Alive (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Journey - Don\'t Stop Believin\' (Live 1981: Escape Tour - 2022 HD Remaster).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Gary Glitter - Rock & Roll Part II | Joker OST.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Black Crowes - She Talks To Angels (Official Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Bon Jovi - You Give Love A Bad Name.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/BLACK SABBATH - "Paranoid" (Official Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Foo Fighters - Everlong (Official HD Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Redbone - Come and Get Your Love (Single Edit - Audio).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Arctic Monkeys - Why\'d You Only Call Me When You\'re High? (Official Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Green Day - American Idiot [Official Music Video] [4K Upgrade].mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/U2 - With Or Without You (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Nickelback - Rockstar [OFFICIAL VIDEO].mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Cure - Friday I\'m In Love.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Brass Monkey.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Ozzy Osbourne - Under the Graveyard (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Radiohead - Creep.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Blackstreet - No Diggity (Official Music Video) ft. Dr. Dre, Queen Pen.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Red Hot Chili Peppers - Under The Bridge [Official Music Video].mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/KONGOS - Come with Me Now.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Police - Walking On The Moon (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Foo Fighters - Best Of You (Official HD Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Sweet Dreams (Are Made of This) (2005 Remaster).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Nirvana - Come As You Are (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/AWOLNATION - Sail (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Cure - Just Like Heaven.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Foo Fighters - Learn To Fly (Official HD Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Milky Chance - Stolen Dance (Official Video) [HD Version].mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Police - Every Breath You Take (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Journey - Faithfully (Official HD Video - 1983).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Cage The Elephant - Come A Little Closer (Official Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Beastie Boys - No Sleep Till Brooklyn (Official Music Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/The Killers - The Man.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Electric Light Orchestra - Mr. Blue Sky (Official Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Blue Swede - Hooked On A Feeling.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Bon Jovi - Livin\' On A Prayer.mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Arctic Monkeys - Do I Wanna Know? (Official Video).mp3',
    '/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/Green Day - Boulevard Of Broken Dreams [Official Music Video] [4K Upgrade].mp3'
]
LYRICS_BASE_DIR = "/home/psyph3ri0n/Documents/projects-2025/sur/fetcher/downloads/Rock playlist/lyrics/"
OUTPUT_MATCHED_CSV = "final_matched_playlist.csv"
OUTPUT_UNMATCHED_CSV = "final_unmatched_playlist.csv"
MATCH_THRESHOLD = 85 # Start higher now, assuming better potential matches

# --- Normalization Function (from v3) ---
def generate_alias_v3(raw_title):
    if not isinstance(raw_title, str): return ""
    name = unidecode(raw_title.lower())
    name = re.sub(r"[\(\[].*?[\)\]]", "", name)
    name = name.replace('-', ' ').replace('–', ' ').replace('/', ' ').replace('&', 'and').replace('•',' ').replace('~',' ')
    keywords_to_remove = [
        r'\bremaster(ed)?\s?(\d{4})?\b', r'\banniversary\s?(version|mix)?\b',
        r'\b(official|audio|video|lyric|visualizer)\b', r'\b(version|mix|edit|cut|live|mono|stereo|album|single|radio)\b',
        r'\b(feat|ft|with)\b', r'\b(original)\b', r'\bdemo\b', r'\bpt\b', r'\bpart\b',
        r'\b\d{4}\b', r'^\d+\s*[\.\)]?\s*', r'\d+"', r'#\d+', r'\|.*?ost', # Added pipe removal for Joker OST
    ]
    for pattern in keywords_to_remove: name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    # Specific common substitutions can go here if needed
    # name = name.replace("ac dc", "acdc")
    return name

# --- Main Script ---
print("--- Playlist Matching Script ---")

# 1. Load Playlist Data
try:
    with open(PLAYLIST_JSON_PATH, 'r', encoding='utf-8') as f:
        playlist_data = json.load(f)
    print(f"Loaded {len(playlist_data)} songs from playlist '{PLAYLIST_JSON_PATH}'.")
except FileNotFoundError:
    print(f"ERROR: Playlist JSON file not found at '{PLAYLIST_JSON_PATH}'")
    exit()
except json.JSONDecodeError:
    print(f"ERROR: Could not decode JSON from '{PLAYLIST_JSON_PATH}'. Check formatting.")
    exit()
except Exception as e:
    print(f"ERROR: Failed to load playlist JSON: {e}")
    exit()

# 2. Build File Alias Lookup from ACTUAL_SONG_PATHS
file_alias_to_path = {}
processed_file_aliases = set()
print(f"Processing {len(ACTUAL_SONG_PATHS)} actual song file paths...")

for file_path in ACTUAL_SONG_PATHS:
    if not isinstance(file_path, str) or not file_path.lower().endswith('.mp3'):
        print(f"WARNING: Skipping invalid or non-MP3 path: {file_path}")
        continue
    filename = os.path.basename(file_path)
    filename_base = os.path.splitext(filename)[0]
    file_alias = generate_alias_v3(filename_base)

    if not file_alias:
        print(f"WARNING: Generated empty alias for file: {filename}. Skipping.")
        continue

    # Handle potential duplicate aliases - keep the first one encountered
    if file_alias in processed_file_aliases:
        # print(f"INFO: Duplicate file alias '{file_alias}' detected. Keeping first occurrence: {file_alias_to_path[file_alias]}")
        continue
    else:
        file_alias_to_path[file_alias] = file_path
        processed_file_aliases.add(file_alias)

available_file_aliases = list(file_alias_to_path.keys())
print(f"Created lookup for {len(available_file_aliases)} unique file aliases.")
if not available_file_aliases:
    print("ERROR: No valid file aliases could be generated. Cannot match.")
    exit()

# 3. Perform Matching
matched_records = []
unmatched_records = []
used_file_paths = set()

print(f"Starting matching process (Threshold: {MATCH_THRESHOLD})...")
for i, song in enumerate(playlist_data):
    if not song or not song.get("track") or not song.get("artist"):
        print(f"WARNING: Skipping playlist item {i+1} due to missing 'track' or 'artist': {song}")
        unmatched_records.append({**song, "match_reason": "Missing Artist/Track in Playlist Data"})
        continue

    track = str(song["track"])
    artist = str(song["artist"])
    playlist_display_title = f"{artist} - {track}" # For logging

    playlist_alias = generate_alias_v3(f"{artist} {track}")
    if not playlist_alias:
         print(f"WARNING: Skipping playlist item '{playlist_display_title}' due to empty generated alias.")
         unmatched_records.append({**song, "match_reason": "Empty Playlist Alias"})
         continue

    # Find the best match using token_set_ratio
    match_result = process.extractOne(playlist_alias, available_file_aliases, scorer=fuzz.token_set_ratio, score_cutoff=MATCH_THRESHOLD)

    matched_audio_path = None
    matched_lyrics_path = None
    match_score = 0
    match_reason = ""

    if match_result:
        best_file_alias, score = match_result[0], match_result[1]
        potential_audio_path = file_alias_to_path.get(best_file_alias)

        if potential_audio_path:
            if potential_audio_path in used_file_paths:
                match_reason = f"File already matched to another song (Score: {score}, File: {os.path.basename(potential_audio_path)})"
                # Still add to unmatched, don't reuse file
            else:
                # Successful Match!
                matched_audio_path = potential_audio_path
                match_score = score
                match_reason = f"Matched (Score: {score})"
                used_file_paths.add(matched_audio_path)

                # Infer and check lyrics path
                audio_filename_base = os.path.splitext(os.path.basename(matched_audio_path))[0]
                potential_lyrics_filename = f"{audio_filename_base}.txt" # Simple .txt extension
                # More robust: Sanitize the base name for lyrics file if needed
                # potential_lyrics_filename = f"{sanitize_for_lyrics(audio_filename_base)}.txt"
                inferred_lyrics_path = os.path.join(LYRICS_BASE_DIR, potential_lyrics_filename)

                if os.path.exists(inferred_lyrics_path):
                    matched_lyrics_path = inferred_lyrics_path
                    # print(f"INFO: Found corresponding lyrics file: {inferred_lyrics_path}")
                else:
                    # print(f"INFO: Could not find inferred lyrics file: {inferred_lyrics_path}")
                    pass # Keep matched_lyrics_path as None
        else:
            # This shouldn't happen if the dictionary is built correctly
            match_reason = f"Internal Error: Matched alias '{best_file_alias}' not found in lookup."
            print(f"ERROR: {match_reason} for '{playlist_display_title}'")

    else:
        # No match found above threshold
        best_guess, best_score = process.extractOne(playlist_alias, available_file_aliases, scorer=fuzz.token_set_ratio) or (None, 0)
        best_guess_file = "N/A"
        if best_guess:
            best_guess_file = os.path.basename(file_alias_to_path.get(best_guess, "N/A"))
        match_reason = f"No match >= {MATCH_THRESHOLD} (Best guess: '{best_guess_file}' with score {best_score})"


    # Add to appropriate list
    if matched_audio_path:
         record = {
            **song, # Include all original playlist data
            "audio_path": matched_audio_path,
            "lyrics_path": matched_lyrics_path, # Will be None if not found
            "match_score": match_score,
            "match_reason": match_reason,
            "playlist_alias": playlist_alias # Add alias used for matching
         }
         matched_records.append(record)
    else:
        # Add original song data plus the reason for failure
        unmatched_records.append({**song, "match_reason": match_reason, "playlist_alias": playlist_alias})

# 4. Save Results
print("\n--- Saving Results ---")

# Save Matched
if matched_records:
    df_matched = pd.DataFrame(matched_records)
    # Define desired column order
    cols_order_matched = ['artist', 'track', 'match_score', 'audio_path', 'lyrics_path', 'match_reason', 'playlist_alias']
    # Get remaining columns from original data
    original_cols = [col for col in df_matched.columns if col not in cols_order_matched]
    # Combine and reorder
    df_matched = df_matched[cols_order_matched + original_cols]
    df_matched.to_csv(OUTPUT_MATCHED_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ Saved {len(df_matched)} matched records to: '{OUTPUT_MATCHED_CSV}'")
else:
    print("ℹ️ No records were matched successfully.")

# Save Unmatched
if unmatched_records:
    df_unmatched = pd.DataFrame(unmatched_records)
     # Define desired column order
    cols_order_unmatched = ['artist', 'track', 'match_reason', 'playlist_alias']
     # Get remaining columns from original data
    original_cols_unmatched = [col for col in df_unmatched.columns if col not in cols_order_unmatched]
    # Combine and reorder
    df_unmatched = df_unmatched[cols_order_unmatched + original_cols_unmatched]
    df_unmatched.to_csv(OUTPUT_UNMATCHED_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"ℹ️ Saved {len(df_unmatched)} unmatched records to: '{OUTPUT_UNMATCHED_CSV}'")
else:
    print("🎉 All playlist records were matched!") # Unlikely given input counts

print("\n--- Analysis & Next Steps ---")
total_playlist = len(playlist_data)
num_matched = len(matched_records)
num_unmatched = len(unmatched_records)
num_available_files = len(ACTUAL_SONG_PATHS)

print(f"Total Playlist Songs: {total_playlist}")
print(f"Available MP3 Files: {num_available_files}")
print(f"Successfully Matched: {num_matched}")
print(f"Unmatched: {num_unmatched}")

print("\nRecommendations:")
print(f"1. Review '{OUTPUT_UNMATCHED_CSV}'. Many failures are likely due to missing MP3 files.")
print(f"   - Identify songs listed there that you *do* have MP3s for but weren't matched.")
print(f"   - Check their 'playlist_alias' vs. the alias of the actual file (you can manually run generate_alias_v3 on the filename).")
print(f"   - Filenames missing artist names (like '505.mp3', 'The Unforgiven (Remastered).mp3') are prime candidates for manual renaming to 'Artist - Title.mp3'.")
print(f"2. Consider renaming inconsistent files you *do* have for better future matching.")
print(f"3. Acquire the MP3s for the ~{total_playlist - num_available_files} missing songs if you want to match the full playlist.")
print(f"4. Manually verify the 'lyrics_path' column in '{OUTPUT_MATCHED_CSV}'. The script inferred paths, but check they contain the correct lyrics.")
print(f"5. If match quality is still low for files you *know* should match, you could try slightly lowering MATCH_THRESHOLD (e.g., to 80 or 75) and rerunning, but examine results carefully.")
print("--- Script Finished ---")