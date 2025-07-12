import os
import pickle
import pandas as pd
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

#Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
#-----------------------------------------------------



# Load or create cached data for efficiency
@st.cache_data
def preprocess_and_cache_data():
    if not os.path.exists("df.pkl") or not os.path.exists("similarity.pkl"):

        # Load original CSV data
        df = pd.read_csv("spotify_millsongdata.csv")
        df = df[['song', 'artist', 'text']].dropna().reset_index(drop=True)

        # Sample only 5000 rows to avoid memory issues
        df = df.sample(5000, random_state=42).reset_index(drop=True)

        # Convert data types to strings
        df['song'] = df['song'].astype(str)
        df['artist'] = df['artist'].astype(str)
        df['text'] = df['text'].astype(str)

        # Compute TF-IDF matrix for lyrics
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['text'])
        
        # Calculate cosine similarity between songs
        similarity = cosine_similarity(tfidf_matrix)
        
        # Cache the processed data
        with open('df.pkl', 'wb') as f:
            pickle.dump(df, f)

        with open('similarity.pkl', 'wb') as f:
            pickle.dump(similarity, f)

        print("✅ similarity.pkl saved.")

    else:
        df = pickle.load(open('df.pkl', 'rb'))
        similarity = pickle.load(open('similarity.pkl', 'rb'))

    return df, similarity


# Load data
music, similarity = preprocess_and_cache_data()


# Validation check
if music is None or similarity is None or not isinstance(music, pd.DataFrame):
    st.error("❌ Failed to load music data or similarity matrix.")
    st.stop()




# Get Spotify track ID for linking
def get_song_album_cover_url(song_name, artist_name):
    try:
        search_query = f"track:{song_name} artist:{artist_name}"
        results = sp.search(q=search_query, type="track")

        if results and results["tracks"]["items"]:
            track = results["tracks"]["items"][0]
            return track["album"]["images"][0]["url"]
    except:
        pass

    # Fallback placeholder
    return "https://i.postimg.cc/0QNxYz4V/social.png"

#mini-player embedding
def get_spotify_track_id(song_name, artist_name):
    try:
        search_query = f"track:{song_name} artist:{artist_name}"
        results = sp.search(q=search_query, type="track")

        if results and results["tracks"]["items"]:
            return results["tracks"]["items"][0]["id"]  # Spotify Track ID
    except:
        pass
    return None

# Generate recommendations using similarity matrix
def recommend(song_name):
    matches = music[music['song'].str.lower() == song_name.lower()]
    if matches.empty:
        return [], []

    idx = matches.index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

    recommended_music_names = []
    recommended_music_posters = []
    for i in distances[1:6]:
        artist = music.iloc[i[0]].artist
        title = music.iloc[i[0]].song
        recommended_music_names.append(title)
        recommended_music_posters.append(get_song_album_cover_url(title, artist))

    return recommended_music_names, recommended_music_posters

# Streamlit UI setup
st.set_page_config(page_title="Music Recommender", layout="wide")

st.markdown("""
<style>
.card:hover {
    transform: scale(1.03);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    transition: all 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("""
<h1 style='text-align: center; color: #1DB954;'>
🎧 Intelligent Music Selector
</h1>
""", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Discover lyrics-inspired musical vibes 🎶</p>", unsafe_allow_html=True)




music_list = music['song'].drop_duplicates().sort_values().values
selected_song = st.selectbox("Type or select a song from the dropdown", music_list)



# On click, show recommendations
if st.button("Show Recommendation"):
    names, posters = recommend(selected_song)
    if not names:
        st.warning("No recommendations found.")
    else:
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                title = names[idx]
                artist = music[music['song'] == title]['artist'].values[0]
                track_id = get_spotify_track_id(title, artist)
                spotify_url = f"https://open.spotify.com/track/{track_id}" if track_id else "#"

                st.markdown(f"""
                    <div class="card" style="background-color:#1e1e1e;padding:10px;border-radius:15px;text-align:center;">
                        <p style="color:white;font-weight:bold;">{title}</p>
                        <img src="{posters[idx]}" width="100%" style="border-radius:10px;" />
                        {f'<a href="https://open.spotify.com/track/{track_id}" target="_blank" style="color:#1DB954;font-weight:bold;">▶ Play on Spotify</a>' if track_id else ""}

                """, unsafe_allow_html=True)

                

    

