# -*- coding: utf-8 -*-
"""
Model Loader - Singleton pattern for loading models once.
Models are cached after first load to avoid reloading on every request.
"""

import os
import pickle
import logging
from typing import Optional

log = logging.getLogger(__name__)

# Global model cache
_model_cache = {}


class ModelLoader:
    """
    Singleton model loader that caches models after first load.
    
    Usage:
        loader = ModelLoader.get_instance()
        dense_model = loader.get_dense_model()
        bm25_encoder = loader.get_bm25_encoder()
    """
    
    _instance = None
    
    def __init__(self):
        self._dense_model = None
        self._bm25_encoder = None
        self._qdrant_client = None
        self._models_loaded = False
        
        # Config paths
        self._script_dir = os.path.dirname(os.path.abspath(__file__))
        self._bm25_path = os.path.join(self._script_dir, 'bm25_model.pkl')
    
    @classmethod
    def get_instance(cls) -> 'ModelLoader':
        """Get singleton instance of ModelLoader."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are already loaded."""
        return self._models_loaded
    
    def get_dense_model(self, model_name: str = None):
        """
        Get dense embedding model (SentenceTransformer).
        Loads from cache if available.
        """
        if self._dense_model is not None:
            return self._dense_model
        
        from sentence_transformers import SentenceTransformer
        
        if model_name is None:
            try:
                from rag_law.config import DENSE_MODEL_ID
                model_name = DENSE_MODEL_ID
            except ImportError:
                model_name = "Savoxism/vietnamese-legal-embedding-finetuned"
        
        log.info(f"Loading dense model: {model_name}")
        print(f"[ModelLoader] Loading dense model: {model_name}...")
        
        self._dense_model = SentenceTransformer(model_name)
        print(f"[ModelLoader] Dense model loaded (dim={self._dense_model.get_sentence_embedding_dimension()})")
        
        return self._dense_model
    
    def get_bm25_encoder(self, corpus: list = None):
        """
        Get BM25 sparse encoder.
        Loads from saved file if available, otherwise fits on corpus.
        
        Args:
            corpus: List of texts to fit BM25 on (only needed if no saved model)
        """
        if self._bm25_encoder is not None:
            return self._bm25_encoder
        
        # Try to load from saved file
        if os.path.exists(self._bm25_path):
            log.info(f"Loading BM25 from: {self._bm25_path}")
            print(f"[ModelLoader] Loading BM25 from cache...")
            
            self._bm25_encoder = BM25Encoder()
            self._bm25_encoder.load(self._bm25_path)
            
            print(f"[ModelLoader] BM25 loaded (vocab_size={len(self._bm25_encoder.vocab)})")
            return self._bm25_encoder
        
        # Fit new BM25 if corpus provided
        if corpus is not None:
            log.info("Fitting new BM25 encoder...")
            print(f"[ModelLoader] Fitting BM25 on {len(corpus)} documents...")
            
            self._bm25_encoder = BM25Encoder()
            self._bm25_encoder.fit(corpus)
            self._bm25_encoder.save(self._bm25_path)
            
            print(f"[ModelLoader] BM25 saved to: {self._bm25_path}")
            return self._bm25_encoder
        
        raise ValueError("BM25 model not found and no corpus provided for fitting")
    
    def get_qdrant_client(self, url: str = None, api_key: str = None):
        """
        Get Qdrant client.
        Reuses existing connection if available.
        """
        if self._qdrant_client is not None:
            return self._qdrant_client
        
        from qdrant_client import QdrantClient
        
        if url is None or api_key is None:
            try:
                from rag_law.config import QDRANT_URL, QDRANT_API_KEY
                url = url or QDRANT_URL
                api_key = api_key or QDRANT_API_KEY
            except ImportError:
                url = os.getenv("QDRANT_URL", "")
                api_key = os.getenv("QDRANT_API_KEY", "")
        
        if not url or not api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY required")
        
        log.info(f"Connecting to Qdrant: {url}")
        print(f"[ModelLoader] Connecting to Qdrant...")
        
        self._qdrant_client = QdrantClient(url=url, api_key=api_key)
        print(f"[ModelLoader] Qdrant connected")
        
        return self._qdrant_client
    
    def load_all(self, corpus: list = None):
        """
        Load all models at once.
        Call this at startup to pre-load everything.
        """
        if self._models_loaded:
            print("[ModelLoader] Models already loaded")
            return
        
        print("=" * 50)
        print("[ModelLoader] Loading all models...")
        print("=" * 50)
        
        self.get_dense_model()
        
        if corpus or os.path.exists(self._bm25_path):
            self.get_bm25_encoder(corpus)
        
        self._models_loaded = True
        print("=" * 50)
        print("[ModelLoader] All models loaded!")
        print("=" * 50)
    
    def clear_cache(self):
        """Clear all cached models."""
        self._dense_model = None
        self._bm25_encoder = None
        self._qdrant_client = None
        self._models_loaded = False
        print("[ModelLoader] Cache cleared")


class BM25Encoder:
    """BM25 sparse encoder for Qdrant."""
    
    def __init__(self):
        self.bm25 = None
        self.vocab = {}
    
    def fit(self, corpus: list):
        """Fit BM25 on corpus."""
        from rank_bm25 import BM25Okapi
        import re
        
        def tokenize(text: str) -> list:
            text = text.lower()
            return re.findall(r'\w+', text)
        
        tokenized = [tokenize(doc) for doc in corpus]
        
        # Build vocabulary
        self.vocab = {}
        idx = 0
        for doc_tokens in tokenized:
            for token in doc_tokens:
                if token not in self.vocab:
                    self.vocab[token] = idx
                    idx += 1
        
        self.bm25 = BM25Okapi(tokenized)
        log.info(f"BM25 fitted: {len(corpus)} docs, {len(self.vocab)} vocab")
    
    def encode(self, text: str) -> dict:
        """Encode text to sparse vector."""
        import re
        tokens = re.findall(r'\w+', text.lower())
        
        indices = []
        values = []
        
        for token in set(tokens):
            if token in self.vocab:
                indices.append(self.vocab[token])
                values.append(float(tokens.count(token)))
        
        return {"indices": indices, "values": values}
    
    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'vocab': self.vocab}, f)
    
    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.vocab = data['vocab']


# Convenience functions
def get_loader() -> ModelLoader:
    """Get the global model loader instance."""
    return ModelLoader.get_instance()


def preload_models():
    """Pre-load all models at startup."""
    loader = get_loader()
    loader.load_all()


if __name__ == "__main__":
    # Test model loading
    print("Testing ModelLoader...")
    
    loader = ModelLoader.get_instance()
    
    # First load
    print("\n--- First load ---")
    dense = loader.get_dense_model()
    print(f"Dense model dim: {dense.get_sentence_embedding_dimension()}")
    
    # Second load (should be cached)
    print("\n--- Second load (cached) ---")
    dense2 = loader.get_dense_model()
    print(f"Same instance: {dense is dense2}")
    
    print("\n✅ ModelLoader working correctly!")
