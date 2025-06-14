from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8888
    DEBUG: bool = False

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080"
    ]

    # Database
    DATABASE_URL: str = "sqlite:///./thai_rag.db"
    # DATABASE_URL: str = "postgresql://user:password@localhost/thai_rag"  # For PostgreSQL

    # Vector Database Settings
    VECTOR_DB_TYPE: str = "chroma"  # Options: chroma, pinecone, weaviate
    VECTOR_DB_HOST: str = "localhost"  # For ChromaDB
    VECTOR_DB_PORT: int = 8000  # Default port for ChromaDB
    VECTOR_COLLECTION_NAME: str = "thai_rag_collection"
    CHROMA_PERSIST_DIR: str = ".chroma_db"

    # # Pinecone settings (if using Pinecone)
    # PINECONE_API_KEY: str = ""
    # PINECONE_ENVIRONMENT: str = ""
    # PINECONE_INDEX_NAME: str = "thai-rag-index"

    # Embedding Model Settings
    # EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    # EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"  # Alternative
    EMBEDDING_DEVICE: str = "cuda"  # PC
    # EMBEDDING_DEVICE: str = "cpu"  # or "cuda" if GPU available ## Mac
    EMBEDDING_DIMENSION: int = 1024  # For MiniLM model

    # OCR Settings
    OCR_ENGINE: str = "tesseract"  # easyocr, tesseract
    OCR_LANGUAGES: List[str] = ["th", "en"]  # Thai and English
    OCR_GPU: bool = True
    OCR_USE_GPU_CORRECTOR: bool = False
    OCR_ENABLE_SPELL_CORRECTION: bool = True  # Setting: Enable/Disable spell correction

    OCR_TESSERACT_EXECUTE_LOCATION: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe" ## Homebrew -> "/opt/homebrew/bin/tesseract"
    OCR_TESSERACT_PATH_TO_TRAINED_DATA:str = r"C:\Program Files\Tesseract-OCR\tessdata" ## Windows
    OCR_TESSERACT_LANGUAGE: str = "tha+eng"
    OCR_TESSERACT_CONFIG: str = "-l tha+eng --psm 6"

    OCR_PROCESSING_MAX_MEMORY_GB: int = 4  # Limit to 4 GB memory usage
    OCR_PDF_PROCESSING_MAX_PARALLEL_PAGES: int = 2  # Process 2 pages at a time
    OCR_PDF_PROCESSING_TARGET_DPI: int = 200  # Lower DPI for memory saving
    OCR_PDF_PROCESSING_MEMORY_CHECK_INTERVAL: int = 5  # Check memory every 5 pages

    OCR_PYTHAINLP_TOKENIZE_ENGINE: str = "whitespace+newline"

    # Text Processing Settings
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50
    MAX_CHUNK_SIZE: int = 1000

    # File Upload Settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    UPLOAD_DIR: str = "./uploads"
    TEMP_DIR: str = "./temp"
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".png", ".jpg", ".jpeg", ".txt"]

    # Thai Language Settings
    THAI_TOKENIZER: str = "newmm"  # Options: newmm, longest, deepcut
    WORD_TOKENIZER_ENGINE: str = "attacut"
    REMOVE_STOPWORDS: bool = True
    NORMALIZE_TEXT: bool = True

    # Search Settings
    DEFAULT_SEARCH_LIMIT: int = 5
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.5
    MAX_SEARCH_LIMIT: int = 50

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # OpenAI API (optional, for comparison)
    OPENAI_API_KEY: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
