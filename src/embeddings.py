# src/embeddings.py
from typing import List
import numpy as np
from fastembed import TextEmbedding

class BilingualEmbedder:
    def __init__(self):
        # Using a lightweight, multilingual model that efficiently supports Arabic and English
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5") 
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Convert numpy float32 arrays to standard Python float lists to ensure JSON storage compatibility
        return [np.array(embeddings).astype(float).tolist() for embeddings in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        # Convert the query vector to the same type to ensure matching computation formats
        query_emb = next(self.model.embed([text]))
        return np.array(query_emb).astype(float).tolist()