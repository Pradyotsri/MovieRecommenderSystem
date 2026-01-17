import streamlit as st
import pandas as pd
import numpy as np
import requests
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System using AI/ML")

# ===============================
# API KEY (use Streamlit Secrets)
# ===============================
TMDB_API_KEY="2c0d9fe76097458a01413386e76cb5b9"

try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except KeyError:
    st.error("TMDB API Key not found. Please add it to .streamlit/secrets.toml")
    st.stop()



# ===============================
# HELPER FUNCTIONS
# ===============================
def convert(text):
    """Convert stringified list of dicts to list of names"""
    L = []
    for i in ast.literal_eval(text):
        L.append(i["name"])
    return L

def convert_cast(text):
    """Return top 3 cast members"""
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i["name"])
            counter += 1
        else:
            break
    return L

def fetch_poster(movie_id):
    """Fetch movie poster from TMDB"""
    if not TMDB_API_KEY:
        return None

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    try:
        data = requests.get(url, timeout=5).json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except:
        return None

    return None

# ===============================
# LOAD & PROCESS DATA
# ===============================
@st.cache_resource
def load_data():
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")

    movies = movies.merge(credits, left_on="id", right_on="movie_id")
    movies = movies[["id", "title_x", "overview", "genres", "keywords", "cast", "crew"]]
    movies.dropna(inplace=True)

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(
        lambda x: [i["name"] for i in ast.literal_eval(x) if i["job"] == "Director"]
    )

    movies["tags"] = (
        movies["overview"]
        + " "
        + movies["genres"].apply(lambda x: " ".join(x))
        + " "
        + movies["keywords"].apply(lambda x: " ".join(x))
        + " "
        + movies["cast"].apply(lambda x: " ".join(x))
        + " "
        + movies["crew"].apply(lambda x: " ".join(x))
    )

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    return movies.reset_index(drop=True), similarity

movies, similarity = load_data()

# ===============================
# RECOMMEND FUNCTION
# ===============================
def recommend(movie_title):
    index = movies[movies["title_x"] == movie_title].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    names = []
    posters = []

    for i in distances:
        movie_id = movies.iloc[i[0]].id
        names.append(movies.iloc[i[0]].title_x)
        posters.append(fetch_poster(movie_id))

    return names, posters

# ===============================
# UI
# ===============================
selected_movie = st.selectbox(
    "Select a movie",
    movies["title_x"].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            if posters[i]:
                st.image(posters[i])
            st.caption(names[i])
