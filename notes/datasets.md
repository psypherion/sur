# GTZN dataset
One of the most used datasets for music genre classification—contains labeled songs by genre.

 - genres original - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
 - images original - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
 - 2 CSV files - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.


# MGD: Music Genre Dataset

 Contains Spotify chart data by region, track, artist, and genre info.

 - Genre Networks: Success-based genre collaboration networks
 - Genre Mapping: Genre mapping from Spotify genres to super-genres
 - Artist Networks: Success-based artist collaboration networks
 - Artists: Some artist data
 - Hit Songs: Hit Song data and features
 - Charts: Enhanced data from Spotify Weekly Top 200 Charts


# trebi/music-genres-dataset

List of genres (over 1,400) each with 200 songs, including genre/sub-genre, track names, and artist, available as a ZIP/JSON.

 - 1494 genres
 - each genre contains 200 songs
 - for each song, following attributes are provided:
     -     artist
     -     song name
     -     position within the list of 200 songs
     -     main genre
     -     sub-genres (with popularity count, which could be interpreted as weight of the sub-genre)
     -     tags (every label that is not some existing genre, usually emotions, "My top 10 favourite tracs" etc.; also with popularity count)