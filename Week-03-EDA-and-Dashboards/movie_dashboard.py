import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="MovieLens Dashboard", layout="wide")
st.title("MovieLens Data Analysis Dashboard")

# Load data
data_path = "data/movie_ratings.csv"
df = pd.read_csv(data_path)

# Sidebar filters
st.sidebar.header("Filters")
ages = st.sidebar.multiselect("Select Age(s)", sorted(df["age"].unique()), default=sorted(df["age"].unique()))
genders = st.sidebar.multiselect("Select Gender(s)", sorted(df["gender"].unique()), default=sorted(df["gender"].unique()))
occupations = st.sidebar.multiselect("Select Occupation(s)", sorted(df["occupation"].unique()), default=sorted(df["occupation"].unique()))
genres_list = sorted(set(g for sublist in df["genres"].str.split("|") for g in sublist))
genres_selected = st.sidebar.multiselect("Select Genre(s)", genres_list, default=genres_list)

# Filter data
df_filtered = df[
    df["age"].isin(ages) &
    df["gender"].isin(genders) &
    df["occupation"].isin(occupations)
]

def genre_explode(df):
    return df.assign(genres=df["genres"].str.split("|")).explode("genres")

df_genre = genre_explode(df_filtered)
df_genre = df_genre[df_genre["genres"].isin(genres_selected)]

st.subheader("Dataset Overview")
st.dataframe(df_filtered.head(100))

# 1. Genre breakdown
st.header("1. Breakdown of Genres for Rated Movies")
genre_counts = df_genre["genres"].value_counts()
st.bar_chart(genre_counts)
st.caption("Number of ratings per genre (movies may belong to multiple genres)")

# 2. Highest viewer satisfaction by genre
st.header("2. Genres with Highest Viewer Satisfaction")
genre_mean_rating = df_genre.groupby("genres")["rating"].mean().sort_values(ascending=False)
st.bar_chart(genre_mean_rating)
st.caption("Mean rating per genre")

# 3. Mean rating across movie release years
st.header("3. Mean Rating Across Movie Release Years")
year_mean_rating = df_filtered.groupby("year")["rating"].mean()
st.line_chart(year_mean_rating)
st.caption("Mean rating by movie release year")

# 4. Top 5 best-rated movies (≥50 and ≥150 ratings)
st.header("4. Top 5 Best-Rated Movies")
movie_counts = df_filtered.groupby("title")["rating"].agg(["count", "mean"])
top_50 = movie_counts[movie_counts["count"] >= 50].sort_values("mean", ascending=False).head(5)
top_150 = movie_counts[movie_counts["count"] >= 150].sort_values("mean", ascending=False).head(5)
st.subheader("Movies with ≥50 Ratings")
st.dataframe(top_50)
st.subheader("Movies with ≥150 Ratings")
st.dataframe(top_150)


