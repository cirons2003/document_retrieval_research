import hdbscan
import numpy as np
import random
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
from typing import List
from ...helpers.embeddings import Embeddings

class HDBSCANClusteringAgent:
    def __init__(self, min_cluster_size: int = 10, umap_model=None, reduced_embeddings=None):
        """
        Initializes the HDBSCANClusteringAgent.

        Args:
            min_cluster_size (int): Minimum number of points to form a cluster.
            umap_model: Pretrained UMAP model (if available).
        """
        self.name = 'HDBSCAN'
        self.min_cluster_size = min_cluster_size
        self.hdbscan = None  # Will be initialized after training
        self._is_trained = False
        self._embeddings = None

        self.umap = umap_model
        self.reduced_embeddings = reduced_embeddings

    def pass_embeddings(self, embeddings: Embeddings):
        """
        Passes the filled Embeddings object to the clustering agent.

        Args:
            embeddings (Embeddings): The embeddings object used for clustering.
        """
        self._embeddings = embeddings

    def train(self):
        """
        Trains HDBSCAN using UMAP-reduced embeddings.
        """
        if self._embeddings is None:
            raise Exception("Embeddings must be passed before training.")
        
        # Reduce dimensionality using UMAP
        all_embeddings = self._embeddings.getAllEmbeddings()
        reduced_embeddings = None
        if self.umap is None:
            self.umap = umap.UMAP(n_components=50, metric='cosine', transform_mode='embedding')      
            reduced_embeddings = self.umap.fit_transform(all_embeddings) 
        elif self.reduced_embeddings is None:
            reduced_embeddings = self.umap.transform(all_embeddings)
        else: 
            reduced_embeddings = self.reduced_embeddings
        # Normalize embeddings so Euclidean distance approximates cosine similarity
        normalized_embeddings = normalize(reduced_embeddings, norm='l2', axis=1)

        # Train HDBSCAN
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, prediction_data=True, metric='euclidean')
        self.hdbscan.fit(normalized_embeddings)

        self._is_trained = True

    def generate_result(self, person_embeddings: List[List[float]]) -> List[int]:
        """
        Generates a list of cluster assignments for a given set of embeddings.

        Args:
            person_embeddings (List[List[float]]): A batch of person embeddings.

        Returns:
            List[int]: Binary vector indicating topic membership.
        """
        if not self._is_trained:
            raise Exception("Train must be called before generating results.")

        # Apply UMAP to new embeddings
        reduced_embeddings = self.umap.transform(person_embeddings)

        # Normalize embeddings
        normalized_embeddings = normalize(reduced_embeddings, norm='l2', axis=1)

        # Predict clusters
        cluster_assignments, _ = hdbscan.approximate_predict(self.hdbscan, normalized_embeddings)

        # Generate a binary vector indicating topic membership
        result_vector = [0] * (max(self.hdbscan.labels_) + 1)   # Number of discovered clusters
        for cluster in cluster_assignments:
            if cluster != -1:  # Ignore noise points
                result_vector[cluster] = 1

        return result_vector

    def topic_map(self, embedding: List[float]) -> int:
        """
        Returns the topic (cluster) corresponding to a single embedding.

        Args:
            embedding (List[float]): A single embedding vector.

        Returns:
            int: Assigned cluster.
        """
        if not self._is_trained:
            raise Exception("Train must be called before topic mapping.")

        # Apply UMAP transformation
        reduced_embedding = self.umap.transform([embedding])
        
        # Normalize
        normalized_embedding = normalize(reduced_embedding, norm='l2', axis=1)

        return int(hdbscan.approximate_predict(self.hdbscan, normalized_embedding)[0][0])

    def is_finished_training(self) -> bool:
        """
        Returns whether the training is complete.

        Returns:
            bool: True if the model is trained, False otherwise.
        """
        return self._is_trained

    def getStats(self, batch_size: int = 20000):
        """
        Computes clustering statistics using a sampled batch of embeddings.

        Args:
            batch_size (int): Number of embeddings to sample for computing metrics.

        Returns:
            dict: A dictionary with clustering statistics.
        """
        if not self._is_trained:
            raise Exception("Train must be called before computing stats.")

        # Get a sample batch of embeddings
        epoch = random.randint(0, 100)
        _, sample_embeddings = self._embeddings.getEmbeddingBatch(0, batch_size, epoch)

        # Ensure we have enough samples
        if len(sample_embeddings) < 2:
            return {
                "Silhouette Score (Separation + Compactness)": -1,
                "Davies-Bouldin Index (Separation)": -1,
                "Calinski-Harabasz Score (Compactness vs Separation)": -1,
                "Cluster Counts": {i: 0 for i in set(self.hdbscan.labels_)}
            }
        
        reduced_sample = self.umap.transform(sample_embeddings)
        normalized_sample = normalize(reduced_sample, norm='l2', axis=1)

        # Predict clusters
        sample_labels, _ = hdbscan.approximate_predict(self.hdbscan, normalized_sample)

        # Remove noise points (-1) for silhouette score
        valid_indices = sample_labels != -1
        valid_embeddings = np.array(sample_embeddings)[valid_indices]
        valid_labels = np.array(sample_labels)[valid_indices]

        # Compute metrics only on valid clusters
        if len(valid_labels) > 1:
            silhouette = silhouette_score(valid_embeddings, valid_labels)
            davies_bouldin = davies_bouldin_score(valid_embeddings, valid_labels)
            calinski_harabasz = calinski_harabasz_score(valid_embeddings, valid_labels)
        else:
            silhouette, davies_bouldin, calinski_harabasz = -1, -1, -1  # Not enough clusters

        # Compute cluster counts
        unique, counts = np.unique(sample_labels, return_counts=True)
        cluster_counts = dict(zip(unique, counts))

        # Return the stats
        stats = {
            "Silhouette Score (Separation + Compactness)": silhouette,
            "Davies-Bouldin Index (Separation)": davies_bouldin,
            "Calinski-Harabasz Score (Compactness vs Separation)": calinski_harabasz,
            "Cluster Counts": cluster_counts
        }

        return stats
