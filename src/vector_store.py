# src/vector_store.py
import json
import os
import numpy as np
from typing import List, Dict, Any

class LocalVectorStore:
    def __init__(self, storage_path: str = "data/vector_store.json"):
        self.storage_path = storage_path
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self.load_store()

    def load_store(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data.get("documents", [])
                self.embeddings = data.get("embeddings", [])

    def save_store(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump({"documents": self.documents, "embeddings": self.embeddings}, f, ensure_ascii=False, indent=4)

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
        for text, meta, emb in zip(texts, metadatas, embeddings):
            self.documents.append({"text": text, "metadata": meta})
            self.embeddings.append(emb)
        self.save_store()

    def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Dict[str, Any]]:
        if not self.embeddings:
            return []

        # Convert matrices for computations via numpy
        query_arr = np.array(query_embedding)
        embeddings_arr = np.array(self.embeddings)

        # Compute Cosine Similarity
        dot_product = np.dot(embeddings_arr, query_arr)
        norm_embeddings = np.linalg.norm(embeddings_arr, axis=1)
        norm_query = np.linalg.norm(query_arr)
        
        similarities = dot_product / (norm_embeddings * norm_query + 1e-8)
        
        # Sort all indices from highest similarity to lowest
        all_sorted_indices = np.argsort(similarities)[::-1]

        results = []
        seen_pages = set() # Set to track retrieved page numbers and prevent duplication

        for idx in all_sorted_indices:
            metadata = self.documents[idx]["metadata"]
            page_num = metadata.get("page") # Extract page number

            # Strict diversity condition: only take the chunk if it belongs to a page not yet captured
            if page_num not in seen_pages:
                seen_pages.add(page_num)
                results.append({
                    "text": self.documents[idx]["text"],
                    "metadata": metadata,
                    "score": float(similarities[idx])
                })
            
            # Once the requested number of unique, diverse pages is reached, break immediately
            if len(results) == k:
                break
                
        return results