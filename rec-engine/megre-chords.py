import os
import json

def merge_chord_progressions():
    """
    Finds all JSON files in the 'db/' subdirectory, pairs the main data files
    with their corresponding chord progression files, merges them, and cleans up.
    """
    # Define the subdirectory where the JSON files are located
    directory = "db/"
    
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"[ERROR] The directory '{directory}' was not found.")
        print("Please make sure your JSON files are inside a 'db' folder in the same directory as the script.")
        return

    files = os.listdir(directory)
    
    file_pairs = {}

    # --- Step 1: Find all json files and group them ---
    for filename in files:
        if filename.endswith(".json"):
            if filename.endswith("-cp.json"):
                base_name = filename.replace("-cp.json", "")
                if base_name not in file_pairs:
                    file_pairs[base_name] = {}
                file_pairs[base_name]['cp_file'] = filename
            else:
                base_name = filename.replace(".json", "")
                if base_name not in file_pairs:
                    file_pairs[base_name] = {}
                file_pairs[base_name]['main_file'] = filename
    
    print(f"Found {len(file_pairs)} potential pairs to process in the '{directory}' directory.")

    # --- Step 2: Process only the complete pairs ---
    for base_name, pair in file_pairs.items():
        if 'main_file' in pair and 'cp_file' in pair:
            # *** KEY CHANGE: Build the full path for each file ***
            main_filepath = os.path.join(directory, pair['main_file'])
            cp_filepath = os.path.join(directory, pair['cp_file'])
            
            print(f"\nProcessing pair: {main_filepath} and {cp_filepath}")

            try:
                # --- Step 3: Read data from both files using the full path ---
                with open(main_filepath, 'r', encoding='utf-8') as f:
                    main_data = json.load(f)
                
                with open(cp_filepath, 'r', encoding='utf-8') as f:
                    cp_data = json.load(f)

                # --- Step 4: Create lookup map (no changes needed here) ---
                cp_map = {
                    (song['title'].strip().lower(), song['artist'].strip().lower()): song['chord_progression']
                    for song in cp_data
                }
                
                # --- Step 5: Merge data (no changes needed here) ---
                songs_updated = 0
                for song in main_data:
                    song_key = (song['title'].strip().lower(), song['artist'].strip().lower())
                    if song_key in cp_map:
                        song['chord_progression'] = cp_map[song_key]
                        songs_updated += 1

                print(f"  > Matched and updated {songs_updated} of {len(main_data)} songs.")

                # --- Step 6: Write back to the main file using the full path ---
                with open(main_filepath, 'w', encoding='utf-8') as f:
                    json.dump(main_data, f, indent=2, ensure_ascii=False)
                
                print(f"  > Successfully updated '{main_filepath}'.")

                # --- Step 7: Remove the chord progression file using the full path ---
                os.remove(cp_filepath)
                print(f"  > Removed '{cp_filepath}'.")

            except json.JSONDecodeError as e:
                print(f"  [ERROR] Could not decode JSON from '{e.doc}'. Please check format. Error: {e}")
            except IOError as e:
                print(f"  [ERROR] Could not read or write file. Error: {e}")
            except Exception as e:
                print(f"  [ERROR] An unexpected error occurred: {e}")
        else:
            if 'main_file' in pair:
                 print(f"\nSkipping '{pair['main_file']}' as it has no matching '-cp.json' file.")

if __name__ == "__main__":
    merge_chord_progressions()
    print("\nScript finished.")