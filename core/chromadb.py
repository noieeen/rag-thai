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

    def add(self, embedding, metadata, id, where=None):
        """
        Add an embedding to the ChromaDB collection.
        """
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[id]
        )

    def search(self, query_embedding, top_k=5, filter_doc_ids=None, threshold=0.0, where=None, include=None):
        """
        Search ChromaDB collection for similar items.
        Supports filter_doc_ids (list of doc_id), threshold (similarity), where clause, and include options.
        """
        where_clause = {}
        if filter_doc_ids:
            where_clause['doc_id'] = {'$in': filter_doc_ids}
        if where:
            where_clause.update(where)
        if not include:
            include = ["metadatas", "distances"]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=include
        )

        filtered = []
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        for metadata, distance in zip(metadatas, distances):
            similarity = 1.0 - distance
            if similarity >= threshold:
                filtered.append({
                    "metadata": metadata,
                    "similarity": similarity
                })
        return filtered