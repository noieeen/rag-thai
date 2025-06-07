import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Simple in-memory vector store for embeddings and associated metadata.
    """

    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.metadatas: List[dict] = []

    def add(self, embedding: np.ndarray, metadata: dict):
        """
        Add an embedding and its metadata to the store.
        """
        self.vectors.append(embedding)
        self.metadatas.append(metadata)
        
        ## Try to print metadata in a more readable format
        print(f"Added embedding with metadata | add: {metadata}")
        if len(self.vectors) % 100 == 0:
            logger.info(f"Added {len(self.vectors)} vectors to the store.")
        else:
            logger.debug(f"Added vector with metadata: {metadata}")
        logger.debug(f"Added vector. Total count: {len(self.vectors)}")

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[Tuple[dict, float]]:
        """
        Search for the top_k most similar embeddings to the query_embedding.
        Optionally filter by similarity threshold and document IDs.
        Returns a list of (metadata, similarity) tuples.
        """
        print(f"Searching for top {top_k} similar vectors. | search")
        
        if not self.vectors:
            logger.warning("Vector store is empty.")
            return []

        similarities = []
        for idx, emb in enumerate(self.vectors):
            sim = self._cosine_similarity(query_embedding, emb)
            
            ## Test
            print(f"Similarity to chunk {idx}: {sim:.4f}")
            meta = self.metadatas[idx]
            if sim >= threshold:
                if filter_doc_ids is None or meta.get("doc_id") in filter_doc_ids:
                    similarities.append((idx, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, sim in similarities[:top_k]:
            results.append((self.metadatas[idx], sim))
        return results

    
    # def hybrid_search(query_embedding, query_tokens):
    #     vector_results = vector_search(...)
    #     bm25_results = bm25_search(...)

    #     # Merge by doc_id or chunk_id
    #     combined_scores = {}
    #     for result in vector_results:
    #         key = result['chunk_id']
    #         combined_scores[key] = combined_scores.get(key, 0) + 0.5 * result['score']

    #     for result in bm25_results:
    #         key = result['chunk_id']
    #         combined_scores[key] = combined_scores.get(key, 0) + 0.5 * result['score']

    #     # Return sorted
    #     return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        if vec1 is None or vec2 is None:
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def clear(self):
        """
        Clear all stored vectors and metadata.
        """
        self.vectors.clear()
        self.metadatas.clear()
        logger.info("Vector store cleared.")

    async def initialize(self):
        """
        Async initialization for vector store (if needed).
        """
        logger.info("VectorStoreService async initialize called.")
        # Add any async setup here if needed
        pass

    def list_documents(self) -> List[dict]:
        """
        List all stored metadata entries.
        """
        return list(self.metadatas)

    async def store_document(self, doc_id: str, filename: str, chunk_embeddings: list):
        """
        Store all chunk embeddings and metadata for a document.
        """
        for chunk in chunk_embeddings:
            metadata = {
                "doc_id": doc_id,
                "filename": filename,
                "text": chunk.text,
                **(chunk.metadata if hasattr(chunk, "metadata") else {})
            }
            self.add(chunk.embedding, metadata)