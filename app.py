
import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favourite movie!")

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id"
)

df = movies[["movie_id", "title_x", "overview", "genres", "keywords", "cast", "crew"]].copy()
df = df.dropna()

def get_names(text):
    items = ast.literal_eval(text)
    return [item["name"] for item in items]

def get_cast(text):
    items = ast.literal_eval(text)
    return [item["name"] for item in items[:3]]

def get_director(text):
    items = ast.literal_eval(text)
    for item in items:
        if item["job"] == "Director":
            return [item["name"]]
    return []

df["genres"] = df["genres"].apply(get_names)
df["keywords"] = df["keywords"].apply(get_names)
df["cast"] = df["cast"].apply(get_cast)
df["crew"] = df["crew"].apply(get_director)

df["overview"] = df["overview"].apply(lambda x: x.split())

df["tags"] = (
    df["overview"]
    + df["genres"]
    + df["keywords"]
    + df["cast"]
    + df["crew"]
)

df["tags"] = df["tags"].apply(lambda x: " ".join(x))
df["tags"] = df["tags"].str.lower()

ps = PorterStemmer()

def stem(text):
    return " ".join(ps.stem(word) for word in text.split())

df["tags"] = df["tags"].apply(stem)

vectorizer = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = vectorizer.fit_transform(df["tags"])
similarity = cosine_similarity(vectors)

titles = df["title_x"].tolist()

def recommend(movie):
    matches = [i for i, title in enumerate(titles) if title.lower() == movie.lower()]

    if not matches:
        st.error("Movie not found. Please choose a movie from the list.")
        return

    index = matches[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    st.subheader("Recommended movies")
    for i, score in scores:
        st.write("🎥", titles[i])

selected_movie = st.selectbox(
    "Choose a movie",
    titles
)

if st.button("Recommend Movies"):
    recommend(selected_movie)
