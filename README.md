# 🎧 Intelligent Music Selector

A lyric-based music recommender system that uses NLP and Spotify integration to suggest similar songs based on the content of their lyrics. Built using Streamlit, scikit-learn, and the Spotify Web API.

---

## 📌 Features

- 🔍 Select any song and get 5 lyrically similar recommendations  
- 🎨 Beautiful UI with album covers and interactive layout  
- 🔗 One-click Spotify links to play recommended tracks  
- ⚡ Fast recommendations using cached similarity matrix  
- 💬 Based purely on lyrics — genre-independent  

---

## 🧠 How It Works

1. Loads a dataset of 237K songs with lyrics (from the [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset)).
2. Samples 5,000 songs for performance.
3. Uses **TF-IDF vectorization** + **cosine similarity** to compute lyrical similarity.
4. Stores preprocessed data and similarity scores using `pickle`.
5. User selects a song → top 5 similar songs are recommended.
6. Spotify API is used to fetch album art and provide external track links.

> ❗Note: Embedded Spotify player was removed due to limitations. Tracks now open in Spotify directly.

---

## 🛠️ Tech Stack

| Layer        | Tools & Libraries                              |
|--------------|------------------------------------------------|
| Frontend     | `Streamlit`, `HTML`, `CSS`                     |
| NLP          | `scikit-learn`, `nltk`, `TfidfVectorizer`      |
| ML Model     | `cosine_similarity` from `sklearn.metrics`     |
| API          | `Spotipy` (Spotify Web API wrapper)            |
| Data         | `spotify_millsongdata.csv`                     |
| Deployment   | Local (Streamlit) or Streamlit Cloud           |

---

## 📁 Project Structure

```bash
music-recommender/
├── main.py                   # Streamlit app
├── Model.ipynb               # Notebook for preprocessing & model building
├── spotify_millsongdata.csv  # Raw dataset of songs + lyrics
├── df.pkl                    # Sampled & cleaned DataFrame
├── similarity.pkl            # Precomputed similarity matrix
├── .venv/ or venv/           # Virtual environment
├── README.md                 # This file
