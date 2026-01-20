import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _build_movie_text(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    tag_agg = (
        tags.dropna(subset=["tag"])
            .assign(tag=lambda d: d["tag"].astype(str).str.lower().str.strip())
            .groupby("movieId")["tag"]
            .apply(lambda s: " ".join(s.value_counts().head(20).index.tolist()))
            .reset_index()
            .rename(columns={"tag": "tagText"})
    )

    df = movies.merge(tag_agg, on="movieId", how="left")
    df["tagText"] = df["tagText"].fillna("")
    df["genresText"] = df["genres"].fillna("").astype(str).str.replace("|", " ", regex=False).str.lower()
    df["text"] = (df["genresText"] + " " + df["tagText"]).str.strip()
    return df

def fit_content_model(movies: pd.DataFrame, tags: pd.DataFrame):
    df = _build_movie_text(movies, tags)
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["text"])
    return df, vectorizer, X

def recommend_similar(title: str, df_movies_text: pd.DataFrame, X, topk: int = 10) -> pd.DataFrame:
    mask = df_movies_text["title"].str.lower().str.contains(title.lower(), regex=False)
    if not mask.any():
        raise ValueError(f"Title not found (try a different substring): {title}")

    idx = df_movies_text[mask].index[0]
    sims = cosine_similarity(X[idx], X).ravel()
    sims[idx] = -1

    top_idx = np.argsort(-sims)[:topk]
    out = df_movies_text.loc[top_idx, ["movieId", "title", "genres"]].copy()
    out["similarity"] = sims[top_idx]
    return out.sort_values("similarity", ascending=False)
