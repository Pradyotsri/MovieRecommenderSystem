import streamlit as st
import pickle
import pandas as pd
import requests

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

TMDB_API_KEY = "2c0d9fe76097458a01413386e76cb5b9"

# ---------------- LOAD DATA ----------------
movies = pickle.load(open("artifacts/movie_list.pkl", "rb"))
similarity = pickle.load(open("artifacts/similarity.pkl", "rb"))

# ---------------- FUNCTIONS ----------------
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()   # raises error for 4xx/5xx
        data = response.json()

        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path

    except requests.exceptions.RequestException as e:
        print("Poster fetch failed:", e)

    # fallback image
    return "https://via.placeholder.com/500x750?text=Poster+Unavailable"


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    names = []
    posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].id
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return names, posters

# ---------------- UI ----------------
st.title("🎬 Movie Recommendation System using AI/ML")

selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values,
    key="movie_selectbox"
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(posters[i])
