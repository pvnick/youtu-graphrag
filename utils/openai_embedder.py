"""
OpenAI-compatible embedding wrapper for llama-server.
Drop-in replacement for SentenceTransformer when using local llama.cpp embedding server.
"""
import os
import numpy as np
import torch
import requests
from typing import List, Union
from dotenv import load_dotenv

load_dotenv()

class OpenAIEmbedder:
    """
    Wrapper that mimics SentenceTransformer interface but uses OpenAI-compatible API.
    Works with llama-server running in embedding mode.
    """
    
    def __init__(self, model_name: str = None, base_url: str = None, api_key: str = None):
        self.base_url = base_url or os.getenv("EMBED_BASE_URL", "http://localhost:8081/v1")
        self.api_key = api_key or os.getenv("EMBED_API_KEY", "not-needed")
        self.model_name = model_name or os.getenv("EMBED_MODEL", "gte-qwen2-7b-instruct")
        self._embedding_dim = None
        
    def encode(
        self, 
        sentences: Union[str, List[str]], 
        convert_to_tensor: bool = False,
        device: str = None,
        batch_size: int = 32,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode sentences to embeddings using OpenAI-compatible API.
        
        Args:
            sentences: Single string or list of strings to encode
            convert_to_tensor: If True, return torch.Tensor instead of numpy array
            device: Target device for tensor (ignored for numpy)
            batch_size: Number of sentences to encode per API call
            
        Returns:
            numpy.ndarray or torch.Tensor of shape (n_sentences, embedding_dim)
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_embeddings = self._encode_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        if convert_to_tensor:
            tensor = torch.from_numpy(embeddings)
            if device:
                tensor = tensor.to(device)
            return tensor
        
        return embeddings
    
    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts via API."""
        url = f"{self.base_url}/embeddings"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            # Sort by index to ensure correct order
            embeddings_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeddings_data]
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding API request failed: {e}")
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self._embedding_dim is None:
            # Encode a test sentence to determine dimension
            test_embedding = self._encode_batch(["test"])[0]
            self._embedding_dim = len(test_embedding)
        return self._embedding_dim


def get_embedder(model_name: str = None):
    """
    Factory function to get appropriate embedder.
    If EMBED_BASE_URL is set, use OpenAI-compatible API.
    Otherwise, fall back to sentence-transformers.
    """
    embed_url = os.getenv("EMBED_BASE_URL")
    
    if embed_url:
        return OpenAIEmbedder(model_name=model_name, base_url=embed_url)
    else:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name or "all-MiniLM-L6-v2")
