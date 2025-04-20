

import librosa
import matplotlib.pyplot as plt
import numpy as np

"""### Sound Waves visualized"""

sound_sample_file = 'data/preview_test.mp3'

# load sound file with librosa
signal, sample_rate = librosa.load(sound_sample_file)

#plot 30 second sample
fig, ax = plt.subplots(nrows=4, figsize=(15,15))
ax[0].set_title('Audio Signal - 30 seconds')
ax[0].set_ylabel('Amplitude')
ax[0].set_xlabel('Time')
ax[0].plot(range(len(signal)), signal)

#plot 1 second sample
signal_sample = signal[:sample_rate]
ax[1].set_title('Audio Signal - 1 seconds')
ax[1].set_ylabel('Amplitude')
ax[1].set_xlabel('Time')
ax[1].plot(range(len(signal_sample)), signal_sample)

#plot 0.1 second sample
signal_short = signal[500:500+sample_rate//10]
ax[2].set_title('Audio Signal - 0.1 second')
ax[2].set_ylabel('Amplitude')
ax[2].set_xlabel('Time')
ax[2].plot(range(len(signal_short)), signal_short)

#plot 0.01 second sample
signal_short = signal[500:500+sample_rate//100]
ax[3].set_title('Audio Signal - 0.01 second')
ax[3].set_ylabel('Amplitude')
ax[3].set_xlabel('Time')
ax[3].plot(range(len(signal_short)), signal_short)

plt.tight_layout()
plt.show()

from skimage.transform import resize

signal, sr = librosa.load(sound_sample_file)

mels = librosa.power_to_db(librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000), ref=np.max)
mel_image = (((80+mels)/80)*255)
mel_image = np.flip(mel_image, axis=0)
mel_image = resize(mel_image, (128,512)).astype(np.uint8)



fig, ax = plt.subplots(nrows=2, figsize=(10,6))
ax[0].set_title('Audio Signal')
ax[0].plot(range(len(signal)), signal)
ax[1].set_title('Mel Spectogram')
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel('Time')
ax[1].imshow(mel_image)

plt.tight_layout()
plt.show()


import pandas as pd
from pyarrow import feather

tracks_df = feather.read_feather('data/all_tracks.feather')
tracks_df = tracks_df.drop_duplicates(subset=['track_id']).reset_index(drop=True)
tracks_df['release_date'] = pd.to_datetime(tracks_df['release_date'],  errors='coerce', infer_datetime_format=True)
tracks_df['year'] = tracks_df['release_date'].dt.year
tracks_df.head()

tracks_df.info()

tracks_df.isna().sum()


missing_links = tracks_df[tracks_df.track_preview_link.isna()]
has_links = tracks_df[~tracks_df.track_preview_link.isna()]

fig, ax = plt.subplots(ncols=3, figsize=(15,5))

ax[0].set_title('Track Popularity')
has_links.track_popularity.hist(ax=ax[0], label='has preview links')
missing_links.track_popularity.hist(ax=ax[0], label='missing links')

ax[1].set_title('Artist Popularity')
has_links.artist_popularity.hist(ax=ax[1], label='has preview links')
missing_links.artist_popularity.hist(ax=ax[1], label='missing links')

ax[2].set_title('Release Year')
has_links.year.hist(ax=ax[2], label='has preview links')
missing_links.year.hist(ax=ax[2], label='missing links')

plt.legend()

print('Has Preview Links - Mean Track Popularity:',round(tracks_df.track_popularity.mean(),2))
print('Missing Links - Mean Track Popularity:',round(missing_links.track_popularity.mean(),2))
print('')
print('Has Preview Links - Mean Artist Popularity:',round(tracks_df.artist_popularity.mean(),2))
print('Missing Links - Mean Artist Popularity:',round(missing_links.artist_popularity.mean(),2))

"""We can see that tracks we are missing links for have higher popularity for each track and each artist associated with that track. That means when we train our neural network on the tracks we will get, there will be a bias towards understanding tracks that are less popular. But there may be other factors we will not be training on as well. For example, what are the distributions of labeled genres for the tracks that don't have preview links?"""

all_genres = set()
for idx, row in tracks_df.iterrows():
    for genre in row.artist_genres:
        all_genres.add(genre)

has_links_genres = dict.fromkeys(all_genres, 0)
for idx, row in has_links.iterrows():
    for genre in row.artist_genres:
        has_links_genres[genre] += 1

missing_links_genres = dict.fromkeys(all_genres, 0)
for idx, row in missing_links.iterrows():
    for genre in row.artist_genres:
        missing_links_genres[genre] += 1

has_links_genres = {k: v for k, v in sorted(has_links_genres.items(), key=lambda item: item[1], reverse=True)}
print('Top 10 genres in tracks that have preview links:')
list(has_links_genres.items())[:10]

missing_links_genres = {k: v for k, v in sorted(missing_links_genres.items(), key=lambda item: item[1], reverse=True)}
print('Top 10 genres in tracks that are missing preview links:')
list(missing_links_genres.items())[:10]


import os
from random import sample
from PIL import Image

base_dir = 'data/Spotify/comp_pngs/'
sample_files = sample(os.listdir(base_dir),12)

images = []
for file in sample_files:
    images.append(Image.open(base_dir + file))

#plot composite image and mel spectorgram, side by side
fig, ax = plt.subplots(ncols=2, figsize=(15,3))
ax[0].set_title('Composite Image: Mel Spectogram, MFCC, Chromagram')
ax[0].imshow(np.array(images[0]))
ax[1].set_title('Mel Spectogram')
ax[1].imshow(np.array(images[0])[:,:,0])

"""Each mel spectrogram contains the frequency and time information for each track and we can see by looking at several examples together that their levels contain a wide range of different images."""

# plot mel spectograms
fig, ax = plt.subplots(ncols=3, nrows=4, figsize=(15,7))
plt.suptitle('Mel Spectograms from different songs')
for i, image in enumerate(images):
    ax[i // 3, i % 3].imshow(np.array(image)[:,:,0])
    ax[i // 3, i % 3].axis('off')
plt.tight_layout()
plt.show()

"""Each image is 512 pixels wide by 128 pixels tall. The neural network will only use images that are 128 by 128 pixels, so here are a few examples of how those slices will appear."""

fig, ax = plt.subplots(ncols=3, figsize=(15,5))
ax[0].imshow(np.array(images[2])[:,:128,0])
ax[1].imshow(np.array(images[5])[:,:128,0])
ax[2].imshow(np.array(images[10])[:,:128,0])




# import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , Flatten, Reshape, Conv2DTranspose, BatchNormalization, Conv1D, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# import custom classes for data generators and saving the latent space after running inference on the entire set.
from src.DataGenerator import AudioDataGenerator
from src.LatentSpace import LatentSpace
from src.helper_functions import plot_reconstruction

# custom data generator initialization

data_gen = AudioDataGenerator(
    directory='data/Spotify/comp_pngs/',
    image_size=(128,512),
    color_mode='rgb',
    batch_size=32,
    sample_size=128000,
    shuffle=True,
    train_test_split=True,
    test_size=0.02,
    output_channel_index=0,
    output_size=(128,128))

img_width = 128
img_height = 128
num_channels=1
kernel_size = 5
strides = 2

# build the auto encoder

class Time_Freq_Autoencoder_Builder:

    def build(width, height, depth, filters=(32,64,128,256), latent_dim=256, kernel_size=5):

        strides = 2

        input_shape = (height, width, depth)
        inputs = Input(shape = input_shape)

        chan_dim = -1

        # input for x_time will be the original image, and x_freq will be the transpose of the image

        x_time = Reshape(target_shape=(height,width))(inputs)
        x_freq = Reshape(target_shape=(height,width))(tf.transpose(inputs, perm=[0,2,1,3]))

        # add Conv1d layers for time encoder

        for f in filters:

            x_time = Conv1D(f, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x_time)
            x_time = BatchNormalization(axis=chan_dim)(x_time)

        # flatten and create a dense layers for half of the latent space dimensions, that will be concatenated with the freq encoder

        x_time = Flatten()(x_time)
        latent_time = Dense(latent_dim//2)(x_time)

        # add Conv1d layers for freq encoder

        for f in filters:

            x_freq = Conv1D(f, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x_freq)
            x_freq = BatchNormalization(axis=chan_dim)(x_freq)

        # flatten and dense layer for frequency latent space

        x_freq = Flatten()(x_freq)
        latent_freq = Dense(latent_dim//2)(x_freq)

        # concatenate the two latent spaces from the two encoders

        latent_concat = tf.keras.layers.Concatenate(axis=1)([latent_time, latent_freq])

        # build encoder from layers

        encoder = Model(inputs, latent_concat, name='encoder')

        # build decoder

        latent_inputs = Input(shape=((latent_dim//2)*2))

        # reshape for expansion with Conv2dTranspose layers

        x = Dense(16384, activation='relu')(latent_inputs)
        x = Reshape(target_shape=(8,8,256))(x)

        # Conv2dTranspose layers

        for f in filters[::-1]:

            x = Conv2DTranspose(f, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
            x = BatchNormalization(axis=chan_dim)(x)

        x = Conv2DTranspose(depth, kernel_size=kernel_size, padding='same', activation='sigmoid')(x)

        outputs = x

        # build decoder from layers

        decoder = Model(latent_inputs, outputs, name='decoder')

        # build autoencoder from encoder and decoder outputs

        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')

        return (encoder, decoder, autoencoder)


# custom Model class to build autoencoder

class Time_Freq_Autoencoder(tf.keras.Model):

    def __init__(self, image_width, image_height, image_depth=1, latent_dim=256, kernel_size=5):
        super().__init__()

        self.encoder, self.decoder, self.autoencoder = Time_Freq_Autoencoder_Builder.build(width=image_width, height=image_height, depth=image_depth, latent_dim=256, kernel_size=kernel_size)

    def call(self, x):
        autoencoded = self.autoencoder(x)
        return autoencoded

autoencoder = Time_Freq_Autoencoder(image_width=img_width, image_height=img_height, latent_dim=256, kernel_size=5)


opt = Adam(learning_rate=1e-3)

autoencoder.compile(optimizer=opt, loss=tf.keras.losses.mse)

autoencoder.build(input_shape=(None,img_height,img_width,num_channels))

autoencoder.summary()


autoencoder.encoder.summary()

autoencoder.decoder.summary()


autoencoder_path = 'data/autoencoder_256dim_time_freq_128k_20epochs'
#autoencoder.save(autoencoder_path)
autoencoder = tf.keras.models.load_model(autoencoder_path)


test_img = data_gen.take(1)[0]
prediction = autoencoder(test_img)

plot_reconstruction(test_img, prediction, 1)


from src.LatentSpace import LatentSpace

latent_space = LatentSpace(autoencoder_path=autoencoder_path,
                        image_dir='data/Spotify/comp_pngs/',
                        tracks_feather_path='data/all_tracks.feather',
                        latent_dims=256,
                        output_size=(128, 128),
                        num_tiles=64)


# latent_space.build()
# latent_space.save(autoencoder_path)

latent_space.load(autoencoder_path)


latent_space.tracks.head()

latent_space.artists.head()

latent_space.genres.head()



import umap
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
import umap.plot
from sklearn.cluster import MiniBatchKMeans
from pyarrow import feather
import matplotlib.pyplot as plt
import seaborn as sns

# isolate the data from only the latent columns for use in the UMAP transformer
data = latent_space.tracks[latent_space.latent_cols]

data


embedding = np.load('data/embedding.npy')
fig, ax = plt.subplots(1, figsize=(10, 10))
plt.scatter(*embedding.T, s=0.3, alpha=.2)

sns.jointplot(*embedding.T, kind='hex', height=10)



from bokeh.resources import INLINE
import bokeh.io

bokeh.io.output_notebook(INLINE)

genre_map = umap.UMAP(n_neighbors=100, min_dist=.1).fit(latent_space.genres[latent_space.latent_cols])
genre_hover_data = pd.DataFrame(latent_space.genres.genre)
umap.plot.output_notebook()
p = umap.plot.interactive(genre_map, point_size=10, hover_data=genre_hover_data)
umap.plot.show(p)


data_sample = latent_space.genres[latent_space.latent_cols]
results=[]
kmeans_predictions = []
gmm_predictions = []
for k in range(2,60):
    kmeans = MiniBatchKMeans(n_clusters=k).fit(data_sample)
    predict = kmeans.predict(data_sample)
    result = {
        'k':k,
        'inertia': kmeans.inertia_,
        'silhouette': silhouette_score(data_sample, predict),
    }
    results.append(result)
    kmeans_predictions.append(predict)
    print(result, end='\r')
cluster_results = pd.DataFrame(results)

clusters = 9

fig, ax = plt.subplots()
ax.set_title('Inertia vs. number of clusters')
cluster_results.inertia.plot(ax=ax)
ax.vlines(x=clusters, ymin=100000, ymax=240000, colors='red', linestyles='dotted')

fig, ax = plt.subplots()
ax.set_title('Silhouette score vs. number of clusters')
cluster_results.silhouette.plot()
ax.vlines(x=clusters, ymin=-.1, ymax=.6, colors='red', linestyles='dotted')



genre_projected = latent_space.genres[latent_space.latent_cols].astype('float32')
kmeans_genre = MiniBatchKMeans(n_clusters=clusters).fit(genre_projected)
p = umap.plot.interactive(genre_map, labels = kmeans_genre.predict(genre_projected), point_size=10, hover_data=genre_hover_data)
umap.plot.show(p)



fig, ax = plt.subplots(1, figsize=(10, 10))
plt.scatter(*embedding.T, s=0.3, alpha=.2, c=kmeans_genre.predict(data), cmap='Spectral')
plt.savefig('img/tracks_clusters_umap.png')
plt.show()



import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# using secret client_id and client_secret stored in json file
f = open('data/apikeys/.apikeys.json')
apikeys = json.load(f)
client_id = apikeys['clientId']
client_secret = apikeys['clientSecret']

credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

# use spotipy to make a Spotify client
spotify = spotipy.Spotify(client_credentials_manager=credentials_manager)

query = 'venus television'

# search Spotify
track = spotify.search(query)['tracks']['items'][0]

print(track['name'])
print(track['artists'][0]['name'])

# save track_id and preview link
track_id = track['id']
link = track['preview_url']

# create mel spectogram and run through the encoder
vector = latent_space.get_vector_from_preview_link(link, track_id)
vector

# get recommendations from LatentSpace cosine similarity search
latents = latent_space.get_similarity(vector, latent_space.tracks, subset=latent_space.latent_cols, num=11)
latents = latents[~latents.track_name.apply(lambda x: x.lower()).isin([track['name'].lower()])][:10].reset_index()


import plotly_express as px
import plotly.offline as pyo
pyo.init_notebook_mode()
from joblib import load

this_track_df = pd.DataFrame(vector, columns=latent_space.latent_cols)
this_track_df['name'] = track['name'] + ' - ' + track['artists'][0]['name']
this_track_df['label'] = 2

latents['name'] = latents['track_name'] + ' - ' + latents['artist_name']
latents['label'] = 1
latents = latents[['name'] + latent_space.latent_cols + ['label']]

base_genres = feather.read_feather('data/base_genres.feather')

genres_and_tracks = pd.concat([base_genres, latents, this_track_df]).reset_index(drop=True)
genre_map = load('data/genre_map.bin')
genre_map_trans = genre_map.transform(genres_and_tracks[latent_space.latent_cols])

genre_map_df = pd.DataFrame(genre_map_trans, columns=['x','y'])
genre_map_df = pd.concat([genres_and_tracks[['name','label']], genre_map_df], axis=1)
genre_map_df.label = genre_map_df.label.map({0:'genre', 1:'similar song', 2:'this song'})
genre_map_df['annotation'] = genre_map_df.apply(lambda x: x['name'] if x['label'] == 'genre' else '', axis=1)

fig = px.scatter(genre_map_df, x='x', y='y', color='label', hover_name='name', size=[.5]*len(genre_map_df), width=800, height=600, text='annotation')
fig.show()



