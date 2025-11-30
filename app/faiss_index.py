import faiss
import numpy as np
import pickle
import os
from app.config import settings

class FaissIndex:
    def __init__(self, dimension=128):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.user_ids = []
        self.index_path = settings.FAISS_INDEX_PATH
        self.load_index()
    
    def add_embedding(self, user_id: str, embedding: np.ndarray):
        self.index.add(embedding.reshape(1, -1))
        self.user_ids.append(user_id)
        self.save_index()
    
    def search(self, embedding: np.ndarray, k=1):
        # Return empty if index is empty
        if self.index.ntotal == 0:
            return []
        
        distances, indices = self.index.search(embedding.reshape(1, -1), k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Check if index is valid (not -1 which means no result)
            if idx >= 0 and idx < len(self.user_ids):
                results.append((self.user_ids[idx], float(dist)))
        return results
    
    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(f"{self.index_path}.meta", "wb") as f:
            pickle.dump(self.user_ids, f)
    
    def load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(f"{self.index_path}.meta", "rb") as f:
                self.user_ids = pickle.load(f)

faiss_index = FaissIndex()
