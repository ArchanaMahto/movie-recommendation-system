import pandas as pd
from .config import MOVIES_CSV, RATINGS_CSV, TAGS_CSV

def load_movies() -> pd.DataFrame:
    return pd.read_csv(MOVIES_CSV)

def load_ratings() -> pd.DataFrame:
    return pd.read_csv(RATINGS_CSV)

def load_tags() -> pd.DataFrame:
    return pd.read_csv(TAGS_CSV)
