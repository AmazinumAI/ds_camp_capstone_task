"""
FAISS-based search index implementation for efficient similarity search.

This module provides a wrapper around FAISS (Facebook AI Similarity Search)
for building and querying vector search indices with inner product similarity.
"""

# Standart library imports
import os
from typing import Tuple

# Thirdparty imports
import faiss

import numpy as np


class FaissSearchHandler:
    """
    A handler class for FAISS-based similarity search operations.

    This class provides an interface for building, querying, saving, and loading
    FAISS indices for efficient vector similarity search using inner product.

    Parameters
    ----------
    dimension : int
        The dimensionality of the vectors to be indexed.

    Attributes
    ----------
    dimension : int
        The dimensionality of the vectors.
    index : faiss.IndexFlatIP
        The FAISS index object for inner product similarity search.
    """

    def __init__(self, dimension: int) -> None:
        """
        Initialize the FAISS search handler.

        Parameters
        ----------
        dimension : int
            The dimensionality of the vectors to be indexed.
        """
        self.dimension: int = dimension
        # Initialize FAISS index with inner product similarity

        self.index = faiss.IndexFlatIP(self.dimension)

    def build(self, embeddings: np.ndarray) -> None:
        """
        Build the search index from a collection of embeddings.

        Adds the provided embeddings to the FAISS index for later querying.

        Parameters
        ----------
        embeddings : np.ndarray
            A 2D array of shape (n_vectors, dimension) containing the vectors
            to be indexed. Each row represents a single vector.
        """
        # Add embeddings to the FAISS index

        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the k most similar vectors to the query embedding.

        Performs a similarity search using inner product distance to find
        the k nearest neighbors to the query vector.

        Parameters
        ----------
        query_embedding : np.ndarray
            A 2D array of shape (n_queries, dimension) containing the query
            vector(s) to search for.
        k : int
            The number of nearest neighbors to return.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - distances : np.ndarray of shape (n_queries, k)
                The similarity scores (inner product values) for each match.
            - indices : np.ndarray of shape (n_queries, k)
                The indices of the k most similar vectors in the original dataset.
        """
        # Perform similarity search and return distances and indices
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def save(self, path: str) -> None:
        """
        Save the FAISS index to disk.

        Persists the current state of the index to a file for later loading.

        Parameters
        ----------
        path : str
            The file path where the index should be saved.
        """
        # Write the FAISS index to the specified file path
        faiss.write_index(self.index, path)

    def load(self, path: str) -> bool:
        """
        Load a previously saved FAISS index from disk.

        Attempts to load an index from the specified file path. If the file
        doesn't exist, the current index remains unchanged.

        Parameters
        ----------
        path : str
            The file path from which to load the index.

        Returns
        -------
        bool
            True if the index was successfully loaded, False if the file
            doesn't exist.
        """
        # Check if the index file exists
        if os.path.exists(path):
            # Load the FAISS index from the file
            self.index = faiss.read_index(path)
            return True
        return False
