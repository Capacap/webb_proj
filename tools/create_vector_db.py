#!/usr/bin/env python3
"""
Create Vector Database - A utility script to create embeddings and a FAISS vector database.

This script processes text files, generates embeddings using spaCy, and creates a searchable
vector database using FAISS for efficient similarity search.
"""

import os
import glob
import pickle
import argparse
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import faiss
import spacy
from tqdm import tqdm
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextChunker:
    """A class to chunk text into smaller pieces for embedding."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of specified size with overlap.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            List of text chunks
        """
        # If text is shorter than chunk_size, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get the chunk from start to start + chunk_size
            end = start + self.chunk_size
            
            # If not at the end of the text, try to find a good break point
            if end < len(text):
                # Look for a period, question mark, or exclamation mark followed by a space
                for i in range(end - 1, start + self.chunk_size // 2, -1):
                    if i < len(text) and text[i] in ['.', '!', '?', '\n'] and (i + 1 == len(text) or text[i + 1].isspace()):
                        end = i + 1
                        break
            
            # Add the chunk to the list
            chunks.append(text[start:end].strip())
            
            # Move start position, considering overlap
            start = end - self.chunk_overlap
            
            # Ensure we're making progress
            if start <= 0:
                start = end
        
        return chunks


class VectorDatabase:
    """A class for creating, saving, and searching a vector database using FAISS."""
    
    def __init__(self, 
                 model_name: str = "en_core_web_lg", 
                 dimension: int = None,
                 index_type: str = "Flat"):
        """
        Initialize the vector database.
        
        Args:
            model_name: The spaCy model to use for embeddings
            dimension: Embedding dimension (if None, determined from model)
            index_type: FAISS index type ('Flat' for exact search, 'IVF' for approximate)
        """
        self.model_name = model_name
        self.index_type = index_type
        
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
        
        # Set dimension based on the model if not provided
        if dimension is None:
            self.dimension = self.nlp("test").vector.shape[0]
        else:
            self.dimension = dimension
            
        logger.info(f"Using embedding dimension: {self.dimension}")
        
        # Initialize FAISS index
        self.index = None
        self.create_empty_index()
        
        # Storage for metadata and texts
        self.metadata = []
        self.texts = []
    
    def create_empty_index(self) -> None:
        """Create an empty FAISS index with the specified parameters."""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            # Use IVF index with 100 centroids (adjust based on dataset size)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            # Note: IVF index needs training with some vectors before use
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        logger.info(f"Created empty FAISS index of type {self.index_type}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text using spaCy."""
        doc = self.nlp(text)
        return doc.vector
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add texts and their metadata to the vector database.
        
        Args:
            texts: List of text strings to embed and add
            metadatas: List of metadata dictionaries for each text
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Check if lengths match
        if len(texts) != len(metadatas):
            raise ValueError("Number of texts and metadata entries must match")
        
        # Generate embeddings for all texts
        embeddings = []
        for text in tqdm(texts, desc="Generating embeddings"):
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Train the index if needed (for IVF)
        if self.index_type == "IVF" and not self.index.is_trained:
            if embeddings_array.shape[0] > 0:
                self.index.train(embeddings_array)
        
        # Add vectors to the index
        self.index.add(embeddings_array)
        
        # Store metadata and texts
        self.metadata.extend(metadatas)
        self.texts.extend(texts)
        
        logger.info(f"Added {len(texts)} texts to the database")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the vector database for texts similar to the query.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        # Generate embedding for the query
        query_embedding = self.get_embedding(query)
        query_embedding_array = np.array([query_embedding]).astype('float32')
        
        # Perform the search
        distances, indices = self.index.search(query_embedding_array, min(k, len(self.metadata)))
        
        # Compile results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue  # Skip invalid indices
                
            result = {
                "score": float(1.0 - dist / 100.0),  # Convert distance to similarity score
                "text": self.texts[idx],
                "metadata": self.metadata[idx]
            }
            results.append(result)
        
        return results
    
    def save(self, directory: str) -> None:
        """
        Save the vector database to disk.
        
        Args:
            directory: Directory to save the database to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save the FAISS index
        index_path = os.path.join(directory, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        # Save texts
        texts_path = os.path.join(directory, "texts.pkl")
        with open(texts_path, 'wb') as f:
            pickle.dump(self.texts, f)
        
        # Save config
        config_path = os.path.join(directory, "config.pkl")
        config = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "index_type": self.index_type
        }
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Saved vector database to {directory}")
    
    @classmethod
    def load(cls, directory: str) -> 'VectorDatabase':
        """
        Load a vector database from disk.
        
        Args:
            directory: Directory to load the database from
            
        Returns:
            Loaded VectorDatabase instance
        """
        # Load config
        config_path = os.path.join(directory, "config.pkl")
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Create instance with loaded config
        instance = cls(
            model_name=config["model_name"],
            dimension=config["dimension"],
            index_type=config["index_type"]
        )
        
        # Load index
        index_path = os.path.join(directory, "index.faiss")
        instance.index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            instance.metadata = pickle.load(f)
            
        # Load texts
        texts_path = os.path.join(directory, "texts.pkl")
        with open(texts_path, 'rb') as f:
            instance.texts = pickle.load(f)
        
        logger.info(f"Loaded vector database from {directory} with {len(instance.metadata)} entries")
        return instance


def create_vector_database(
    wiki_dir: str,
    output_dir: str,
    model_name: str = "en_core_web_lg",
    chunk_size: int = 1000
) -> bool:
    """
    Create a vector database from wiki content.
    
    Args:
        wiki_dir: Directory containing wiki content
        output_dir: Directory to save the vector database
        model_name: Name of the spaCy model to use
        chunk_size: Size of text chunks to process
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize vector database
        vector_db = VectorDatabase(model_name=model_name)
        
        # Process all text files in the wiki directory
        for filename in os.listdir(wiki_dir):
            if not filename.endswith('.txt'):
                continue
                
            file_path = os.path.join(wiki_dir, filename)
            logger.info(f"Processing {filename}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Process content in chunks
            chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                # Create metadata
                metadata = {
                    'source': filename,
                    'chunk_index': i,
                    'title': os.path.splitext(filename)[0],  # Use filename without extension as title
                    'content_preview': chunk[:200] + '...' if len(chunk) > 200 else chunk
                }
                
                # Add to vector database
                vector_db.add_texts([chunk], [metadata])
        
        # Save the vector database
        vector_db.save(output_dir)
        logger.info(f"Successfully created vector database in {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating vector database: {str(e)}")
        return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create a vector database from wiki content using spaCy and FAISS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--wiki-dir",
        default="tools/text_data",
        help="Directory containing wiki text files"
    )
    
    parser.add_argument(
        "--output-dir",
        default="tools/vector_db",
        help="Directory to save the vector database"
    )
    
    parser.add_argument(
        "--model",
        default="en_core_web_lg",
        choices=["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
        help="spaCy model to use for embeddings"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks to process"
    )
    
    args = parser.parse_args()
    
    success = create_vector_database(
        args.wiki_dir,
        args.output_dir,
        args.model,
        args.chunk_size
    )
    
    if success:
        logger.info("Vector database creation completed successfully")
    else:
        logger.error("Failed to create vector database")
        exit(1)


if __name__ == "__main__":
    main() 