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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
                 model_name: str = "en_core_web_md", 
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
        
        # Storage for metadata
        self.metadata = []
    
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
        
        # Store metadata
        self.metadata.extend(metadatas)
        
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
        
        logger.info(f"Loaded vector database from {directory} with {len(instance.metadata)} entries")
        return instance


def parse_file(file_path: str, chunker: TextChunker) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Parse a text file into chunks with metadata.
    
    Args:
        file_path: Path to the text file
        chunker: TextChunker instance for splitting text
        
    Returns:
        List of (text_chunk, metadata) tuples
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the header (expected to be in the format "## Filename")
    header = None
    content_text = content
    header_match = content.split('\n', 1)[0] if '\n' in content else content
    
    if header_match.startswith('##'):
        header = header_match.strip('# \t\n')
        content_text = content.split('\n', 1)[1] if '\n' in content else ""
    
    # Create chunks from the content
    chunks = chunker.chunk_text(content_text)
    
    # Prepare results
    results = []
    for i, chunk in enumerate(chunks):
        # Skip empty chunks
        if not chunk.strip():
            continue
            
        # If we have a header, include it at the start of each chunk
        if header:
            chunk_with_header = f"## {header}\n{chunk}"
        else:
            chunk_with_header = chunk
        
        # Create metadata
        metadata = {
            "source": file_path,
            "chunk_index": i,
            "title": header or os.path.basename(file_path).replace('.txt', ''),
            "content_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
        }
        
        results.append((chunk_with_header, metadata))
    
    return results


def build_vector_database(input_dir: str, 
                          output_dir: str,
                          model_name: str = "en_core_web_md",
                          chunk_size: int = 1000,
                          chunk_overlap: int = 200,
                          index_type: str = "Flat") -> None:
    """
    Build a vector database from text files in the input directory.
    
    Args:
        input_dir: Directory containing text files
        output_dir: Directory to save the vector database to
        model_name: spaCy model to use
        chunk_size: Maximum number of characters per chunk
        chunk_overlap: Number of characters to overlap between chunks
        index_type: FAISS index type ('Flat' or 'IVF')
    """
    # Find all text files
    text_files = glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)
    logger.info(f"Found {len(text_files)} text files in {input_dir}")
    
    if not text_files:
        logger.error(f"No text files found in {input_dir}")
        return
    
    # Create chunker
    chunker = TextChunker(chunk_size, chunk_overlap)
    
    # Create vector database
    vector_db = VectorDatabase(model_name, index_type=index_type)
    
    # Process files
    total_chunks = 0
    for file_path in tqdm(text_files, desc="Processing files"):
        try:
            # Parse file
            chunks_with_metadata = parse_file(file_path, chunker)
            
            if not chunks_with_metadata:
                logger.warning(f"No chunks extracted from {file_path}")
                continue
            
            # Add to vector database
            texts = [chunk for chunk, _ in chunks_with_metadata]
            metadatas = [metadata for _, metadata in chunks_with_metadata]
            
            vector_db.add_texts(texts, metadatas)
            
            total_chunks += len(chunks_with_metadata)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info(f"Processed {len(text_files)} files, created {total_chunks} chunks")
    
    # Save the vector database
    vector_db.save(output_dir)
    logger.info(f"Vector database saved to {output_dir}")


def search_database(db_dir: str, query: str, num_results: int = 5) -> None:
    """
    Search the vector database for similar texts.
    
    Args:
        db_dir: Directory containing the vector database
        query: Search query
        num_results: Number of results to return
    """
    # Load the vector database
    vector_db = VectorDatabase.load(db_dir)
    
    # Perform search
    results = vector_db.search(query, k=num_results)
    
    print(f"\nSearch results for: '{query}'\n")
    print("-" * 80)
    
    for i, result in enumerate(results):
        print(f"[{i+1}] Score: {result['score']:.4f}")
        print(f"Title: {result['metadata']['title']}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Preview: {result['metadata']['content_preview']}")
        print("-" * 80)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create a vector database from text files using spaCy and FAISS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        default="tools/text_data",
        help="Directory containing text files"
    )
    
    parser.add_argument(
        "--output-dir",
        default="tools/vector_db",
        help="Directory to save the vector database"
    )
    
    parser.add_argument(
        "--model",
        default="en_core_web_md",
        choices=["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
        help="spaCy model to use for embeddings"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum number of characters per chunk"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Number of characters to overlap between chunks"
    )
    
    parser.add_argument(
        "--index-type",
        default="Flat",
        choices=["Flat", "IVF"],
        help="FAISS index type (Flat for exact search, IVF for approximate)"
    )
    
    parser.add_argument(
        "--search",
        action="store_true",
        help="Search mode (requires --query)"
    )
    
    parser.add_argument(
        "--query",
        help="Search query (used with --search)"
    )
    
    parser.add_argument(
        "--num-results",
        type=int,
        default=5,
        help="Number of search results to return"
    )
    
    args = parser.parse_args()
    
    if args.search:
        if not args.query:
            parser.error("--search requires --query")
        
        search_database(args.output_dir, args.query, args.num_results)
    else:
        build_vector_database(
            args.input_dir,
            args.output_dir,
            args.model,
            args.chunk_size,
            args.chunk_overlap,
            args.index_type
        )


if __name__ == "__main__":
    main() 