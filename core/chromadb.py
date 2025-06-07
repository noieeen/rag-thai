## Sample ChromaDB Connection and Usage
# import chromadb

# # Connect to your running ChromaDB server (default port 8000)
# client = chromadb.HttpClient(host="localhost", port=8000)

# # Create or get a collection
# collection = client.get_or_create_collection("my_collection")

# # Add embeddings (vectors) with metadata
# collection.add(
#     embeddings=[[0.1, 0.2, 0.3, ...]],  # List of lists (your embedding vectors)
#     metadatas=[{"doc_id": "123", "filename": "example.pdf"}],  # List of dicts
#     ids=["unique_id_1"]  # List of unique string IDs
# )

# # Query for similar vectors
# results = collection.query(
#     query_embeddings=[[0.1, 0.2, 0.3, ...]],  # Query vector(s)
#     n_results=5  # Top K
# )
# print(results)

import chromadb
from core.config import settings


class ChromaVectorStore:
    def __init__(self, host=settings.VECTOR_DB_HOST, port=settings.VECTOR_DB_PORT, collection_name=settings.VECTOR_COLLECTION_NAME):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, embedding, metadata, id):
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[id]
        )

    def search(self, query_embedding, top_k=5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results