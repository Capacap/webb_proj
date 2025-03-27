"""
Vector database implementation using FAISS.
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, TypedDict

import faiss
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchResultMetadata(TypedDict, total=False):
    """Type definition for search result metadata."""
    source: str
    chunk_index: int
    title: str
    content_preview: str

class SearchResult(TypedDict):
    """Type definition for search results."""
    score: float
    metadata: SearchResultMetadata

class VectorStore:
    """
    A vector database implementation using FAISS for similarity search.
    """
    
    def __init__(self, 
                 model_name: str = "en_core_web_lg",
                 vector_db_path: str = None):
        """
        Initialize the vector database.
        
        Args:
            model_name: The spaCy model to use for embeddings
            vector_db_path: Path to the directory containing the vector database files
        """
        self.model_name = model_name
        self.vector_db_path = vector_db_path
        
        # Try to load the spaCy model
        try:
            logger.info(f"Loading spaCy model: {model_name}")
            self.nlp = spacy.load(model_name)
            
            # Disable unnecessary components for faster processing
            self.nlp.disable_pipes(["tagger", "parser", "ner", "lemmatizer"])
            
        except OSError:
            logger.error(f"Model {model_name} not found. Please install it with:")
            logger.error(f"python -m spacy download {model_name}")
            raise
        
        # Set dimension based on the model
        self.dimension = self.nlp("test").vector.shape[0]
        logger.info(f"Using embedding dimension: {self.dimension}")
        
        # Initialize FAISS index and metadata
        self.index = None
        self.metadata = []
        
        # Load the vector database if path is provided
        if vector_db_path:
            self.load(vector_db_path)
        else:
            self._create_empty_index()
    
    def _create_empty_index(self) -> None:
        """Create an empty FAISS index."""
        # Use IVF index for better search quality
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, faiss.METRIC_L2)
        logger.info("Created empty IVF FAISS index")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text using spaCy."""
        doc = self.nlp(text)
        return doc.vector
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query to improve search quality."""
        # Remove common words and normalize
        doc = self.nlp(query.lower())
        # Keep only meaningful words (nouns, verbs, adjectives)
        meaningful_words = [token.text for token in doc 
                          if not token.is_stop and 
                          token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']]
        return " ".join(meaningful_words)
    
    def _rerank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rerank results using additional heuristics."""
        query_doc = self.nlp(query.lower())
        
        def calculate_rerank_score(result: SearchResult) -> float:
            # Get the original similarity score
            base_score = result.get("score", 0.0)
            
            # Get the content
            content = result["metadata"].get("content_preview", "").lower()
            title = result["metadata"].get("title", "").lower()
            
            # Calculate additional scoring factors
            title_match = sum(1 for word in query_doc if word.text in title) / len(query_doc)
            content_match = sum(1 for word in query_doc if word.text in content) / len(query_doc)
            
            # Combine scores with weights
            final_score = (
                base_score * 0.6 +  # Original similarity score
                title_match * 0.2 +  # Title match bonus
                content_match * 0.2   # Content match bonus
            )
            
            return final_score
        
        # Rerank results
        reranked_results = sorted(results, key=calculate_rerank_score, reverse=True)
        return reranked_results
    
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        Search the vector database for texts similar to the query.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        if not self.index or not self.metadata:
            logger.warning("Vector database is empty, no results to return")
            return []
        
        # Preprocess the query
        processed_query = self._preprocess_query(query)
        
        # Generate embedding for the query
        query_embedding = self.get_embedding(processed_query)
        query_embedding_array = np.array([query_embedding]).astype('float32')
        
        # Perform the search with more candidates for reranking
        distances, indices = self.index.search(query_embedding_array, min(k * 2, len(self.metadata)))
        
        # Compile initial results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue  # Skip invalid indices
                
            result: SearchResult = {
                "score": float(1.0 - dist / 100.0),  # Convert distance to similarity score
                "metadata": self.metadata[idx]
            }
            results.append(result)
        
        # Rerank results
        reranked_results = self._rerank_results(results, query)
        
        # Return top k results
        return reranked_results[:k]
    
    def load(self, directory: str) -> bool:
        """
        Load a vector database from disk.
        
        Args:
            directory: Directory to load the database from
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            # Check directory exists
            if not os.path.exists(directory):
                logger.error(f"Vector database directory not found: {directory}")
                return False
                
            # Load metadata
            metadata_path = os.path.join(directory, "metadata.pkl")
            if not os.path.exists(metadata_path):
                logger.error(f"Metadata file not found: {metadata_path}")
                return False
                
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Load index
            index_path = os.path.join(directory, "index.faiss")
            if not os.path.exists(index_path):
                logger.error(f"FAISS index file not found: {index_path}")
                return False
                
            self.index = faiss.read_index(index_path)
            
            logger.info(f"Loaded vector database from {directory} with {len(self.metadata)} entries")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector database: {str(e)}")
            # Reset to empty state on error
            self._create_empty_index()
            self.metadata = []
            return False 