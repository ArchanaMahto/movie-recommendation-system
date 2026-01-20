import argparse
from .data import load_movies, load_ratings, load_tags
from .content_based import fit_content_model, recommend_similar
from .collaborative_svd import train_svd, recommend_for_user

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["content", "cf"], required=True)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--user", type=int, default=1)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    movies = load_movies()
    ratings = load_ratings()

    if args.mode == "content":
        tags = load_tags()
        df_text, _, X = fit_content_model(movies, tags)
        recs = recommend_similar(args.title, df_text, X, topk=args.topk)
        print(recs.to_string(index=False))

    if args.mode == "cf":
        algo, metrics = train_svd(ratings)
        print(f"Eval: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
        recs = recommend_for_user(args.user, algo, movies, ratings, topk=args.topk)
        print(recs.to_string(index=False))

if __name__ == "__main__":
    main()
