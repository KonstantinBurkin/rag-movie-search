from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "movies"

HF_DATASET_SUMMARIES_ID = "vishnupriyavr/wiki-movie-plots-with-summaries"

RERANK_MODEL_NAME = "gemini-3.1-flash-lite"
RERANK_TOP_K = 10
RERANK_CANDIDATE_POOL_SIZE = 30
