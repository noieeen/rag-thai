import chromadb
import numpy as np
from typing import List, Tuple, Optional
import logging
from rank_bm25 import BM25Okapi
import pythainlp
import uuid


logger = logging.getLogger(__name__)

from core.config import settings

bm25_corpus = []
bm25_index = None

class VectorStoreService:
    """
    Simple in-memory vector store for embeddings and associated metadata.
    """

    def __init__(self):
        # Connect to ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        # self.client = chromadb.Client(ChromaSettings(
        #     chroma_db_impl="duckdb+parquet",
        #     persist_directory=settings.CHROMA_PERSIST_DIR
        # ))
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(settings.VECTOR_COLLECTION_NAME)

    def add(self, embedding: np.ndarray, metadata: dict):
        """
        Add an embedding and its metadata to the store.
        """
        # self.vectors.append(embedding)
        # self.metadatas.append(metadata)
        uid = str(uuid.uuid4())
        self.collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            ids=[uid]
        )
        ## Try to print metadata in a more readable format
        print(f"Added embedding with metadata | add: {metadata}")

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[Tuple[dict, float]]:
        query = query_embedding.tolist()
        where_filter = {"doc_id": {"$in": filter_doc_ids}} if filter_doc_ids else None
        results = self.collection.query(
            query_embeddings=[query],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances"]
        )
        output = []
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
            score = 1.0 - distance
            if score >= threshold:
                output.append((metadata, score))
        return output


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
        self.client.delete_collection(settings.VECTOR_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(settings.VECTOR_COLLECTION_NAME)
        logger.info("Vector store cleared.")

    async def close(self):
        """
        Async cleanup for vector store (if needed).
        """
        logger.info("VectorStoreService closed.")
        pass

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
        return self.collection.get(include=["metadatas"])["metadatas"]

    def get_documents(self, document_id: str) -> List[dict]:
        """
        Return all metadata entries for a given document_id.
        """
        results = self.collection.get(
            where={"doc_id": document_id},
            include=["metadatas"]
        )
        return results["metadatas"]

    def delete_documents(self, document_id: str) -> List[dict]:
        """
        Delete all metadata and vectors for a given document_id.
        Returns the list of deleted metadata entries.
        """
        results = self.collection.get(
            where={"doc_id": document_id},
            include=["ids", "metadatas"]
        )
        ids_to_delete = results["ids"]
        self.collection.delete(ids=ids_to_delete)
        return results["metadatas"]

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

    def build_bm25_index(self):
        items = self.collection.get(include=["metadatas"])
        tokenized_corpus = [pythainlp.word_tokenize(m["text"]) for m in items["metadatas"] if "text" in m]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = items["metadatas"]


    def get_collection(self):
        print("Listing all items in Chroma collection:")
        results = self.collection.get(include=["embeddings", "metadatas"])
        for i, (meta, embedding) in enumerate(zip(results['metadatas'], results['embeddings'])):
            print(f"[{i}] Metadata: {meta}\n    Embedding (dim {len(embedding)}): {embedding[:5]}...")