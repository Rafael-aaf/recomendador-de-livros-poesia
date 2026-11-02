from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

BOOKS_PATH = DATA_DIR / "raw" / "goodreads_books_poetry.json"
INTERACTIONS_PATH = DATA_DIR / "raw" / "goodreads_interactions_poetry.json"
REVIEWS_PATH = DATA_DIR / "raw" / "goodreads_reviews_poetry.json"
AUTHORS_PATH = DATA_DIR / "raw" / "goodreads_book_authors.json"