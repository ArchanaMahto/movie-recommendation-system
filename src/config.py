from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ml-latest-small"

MOVIES_CSV = DATA_DIR / "movies.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"
TAGS_CSV = DATA_DIR / "tags.csv"
