import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

def train_svd(ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)

    algo = SVD(random_state=random_state)
    algo.fit(trainset)

    preds = algo.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)
    mae = accuracy.mae(preds, verbose=False)
    return algo, {"rmse": rmse, "mae": mae}

def recommend_for_user(user_id: int, algo, movies: pd.DataFrame, ratings: pd.DataFrame, topk: int = 10) -> pd.DataFrame:
    rated = set(ratings.loc[ratings["userId"] == user_id, "movieId"].tolist())
    candidates = movies.loc[~movies["movieId"].isin(rated), ["movieId", "title", "genres"]].copy()

    est = []
    for mid in candidates["movieId"].tolist():
        est.append(algo.predict(uid=user_id, iid=mid).est)

    candidates["pred_rating"] = est
    return candidates.sort_values("pred_rating", ascending=False).head(topk)
