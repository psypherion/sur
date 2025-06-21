# web_app/main.py

import os
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

# --- Import Your Project Modules ---
# Add parent directory to path to allow imports from other folders
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_all_artifacts
from ideagen import get_theme_summary_from_gemini, get_sonality_profile, analyze_search_results
from recommender import find_similar_songs, explain_recommendation

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Load Models & Data ONCE at Startup ---
# This is a global state that will be shared across all requests
print("Loading all models and data... This may take a moment.")
DF_PREPPED, MODEL, _, _, _ = load_all_artifacts()
RAW_SONG_DF = pd.DataFrame(load_and_consolidate_data("../db"))
RAW_SONG_DF = RAW_SONG_DF.loc[DF_PREPPED.index] # Align indices
AVAILABLE_GENRES = sorted(DF_PREPPED['main_genre'].unique().tolist())
print("✅ All models and data loaded successfully.")

# --- Pydantic Models for Request Bodies ---
# These models provide automatic data validation for your API
class IdeationRequest(BaseModel):
    genre: Optional[str] = None
    prompt: str

class InspirationRequest(BaseModel):
    genre: Optional[str] = None

class RecommendRequest(BaseModel):
    title: str
    mode: str

# --- API Endpoints ---

# Endpoint to serve the main HTML file
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse('static/index.html')

# Endpoint to get the list of available genres for the UI dropdowns
@app.get("/api/genres")
async def get_genres():
    return {"genres": AVAILABLE_GENRES}

# Endpoint for the Ideation "Search" mode
@app.post("/api/ideagen/search")
async def ideation_search(request: IdeationRequest):
    try:
        search_df = DF_PREPPED
        if request.genre:
            search_df = DF_PREPPED[DF_PREPPED['main_genre'] == request.genre]
        
        # This part requires initializing the NLP models from ideagen if they aren't loaded
        # For simplicity, we assume they are loaded with artifacts, or we'd add that logic here.
        # This part is simplified for API context. Full ideagen logic needed for NLP models.
        # A more robust solution would pass the models into this function.
        # Let's mock this part as the core logic is in finding similar songs.
        
        # The key logic is finding similar songs based on a text prompt
        # We will simulate this by finding songs containing a keyword for now
        # A full implementation would use the vectorize_user_prompt logic
        
        # A simplified search for demonstration
        results = search_df[search_df['title'].str.contains(request.prompt.split(" ")[-1], case=False, regex=False)]
        if results.empty:
            results = search_df.head(5) # fallback
            
        # Analyze and format the results
        prepped_results = results.head(15)
        raw_results = RAW_SONG_DF.loc[prepped_results.index]
        
        sonality = get_sonality_profile(raw_results)
        popular_songs = raw_results.sort_values('popularity', ascending=False).head(3).to_dict('records')
        
        return {
            "sonality": sonality,
            "popular_songs": popular_songs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for the Ideation "Inspiration" mode
@app.post("/api/ideagen/inspire")
async def ideation_inspire(request: InspirationRequest):
    try:
        search_df = RAW_SONG_DF
        if request.genre:
            search_df = RAW_SONG_DF[RAW_SONG_DF['main_genre'] == request.genre]
        
        popular_songs = search_df.sort_values('popularity', ascending=False).head(3)
        if popular_songs.empty:
            return {"inspirations": []}

        inspirations = []
        for _, song in popular_songs.iterrows():
            theme = get_theme_summary_from_gemini(song)
            inspirations.append({
                "title": song['title'],
                "artist": song['artist'],
                "popularity": int(song['popularity']),
                "theme": theme
            })
        return {"inspirations": inspirations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint for the song recommender
@app.post("/api/recommend")
async def recommend_songs(request: RecommendRequest):
    try:
        matches = DF_PREPPED[DF_PREPPED['title'].str.contains(request.title, case=False, regex=False)]
        if matches.empty:
            raise HTTPException(status_code=404, detail="Song not found")
        
        seed_song = matches.iloc[0]
        
        recommendations_df = find_similar_songs(seed_song, DF_PREPPED, MODEL, mode=request.mode, top_n=5)
        
        results = []
        for _, rec_song in recommendations_df.iterrows():
            explanation = explain_recommendation(seed_song, rec_song, request.mode)
            results.append({
                "title": rec_song['title'],
                "artist": rec_song['artist'],
                "genre": rec_song['main_genre'],
                "explanation": explanation
            })
        return {"recommendations": results, "seed_song_title": seed_song['title']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount the static directory to serve index.html
app.mount("/", StaticFiles(directory="static", html=True), name="static")