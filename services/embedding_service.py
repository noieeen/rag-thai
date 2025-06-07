import numpy as np
from typing import List, Any
import logging
from sentence_transformers import SentenceTransformer
from core.config import settings


logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating and managing text embeddings using sentence-transformers.
    """
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("EmbeddingService initialized with sentence-transformers.")

    async def initialize(self):
        """
        Async initialization for embedding model (if needed).
        """
        logger.info("EmbeddingService async initialize called.")
        # Add any async setup here if needed
        pass

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text string using sentence-transformers.
        """
        embedding = self.model.encode(text)
        return np.array(embedding)

    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Async wrapper to generate an embedding for a single text string.
        """
        print(f"Generating embedding for text: {text}... | generate_embedding")
        return self.embed_text(text)

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of text strings.
        """
        logger.debug(f"Generating embeddings for {len(texts)} texts.")
        return [self.embed_text(text) for text in texts]

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        """
        if emb1 is None or emb2 is None:
            logger.warning("One or both embeddings are None.")
            return 0.0
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            logger.warning("One or both embeddings have zero norm.")
            return 0.0
        sim = float(np.dot(emb1, emb2) / (norm1 * norm2))
        logger.debug(f"Cosine similarity: {sim}")
        return sim