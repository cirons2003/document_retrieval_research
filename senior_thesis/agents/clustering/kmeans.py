from sklearn.cluster import MiniBatchKMeans
from typing import List
from ...helpers.embeddings import Embeddings
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import random
import matplotlib.pyplot as plt


class KMeansClusteringAgent:
    def __init__(self, topic_count: int):
        """
        Initializes the KMeansClusteringAgent.

        Args:
            name (str): The name of the agent.
            topic_count (int): The number of clusters (topics) to learn.
        """
        self.name = f'KMEANS_{topic_count}'
        self.topic_count = topic_count
        self._kmeans = MiniBatchKMeans(n_clusters=topic_count, batch_size=10000, random_state=42)
        self._is_trained = False
        self._embeddings = None

    def pass_embeddings(self, embeddings: Embeddings):
        """
        Passes the filled Embeddings object to the clustering agent.

        Args:
            embeddings (Embeddings): The embeddings object used for clustering.
        """
        self._embeddings = embeddings

    def train(self):
        """
        Trains MiniBatchKMeans on all embeddings with a progress bar.
        """
        if self._embeddings is None:
            raise Exception("Embeddings must be passed before training.")

        batch_size = 10000
        num_epochs = 20
        for i in range(num_epochs):
            index = 0 
            while True: 
                last_index, batch = self._embeddings.getEmbeddingBatch(index, batch_size, i)
                if len(batch) == 0:
                    break
                self._kmeans.partial_fit(batch)
                index += len(batch) 
                
        self._is_trained = True

    def generate_result(self, person_embeddings: List[List[int]]) -> List[int]:
        """
        Generates a binary topic membership vector for a set of person embeddings.

        Args:
            person_embeddings (List[List[int]]): A list of embeddings for a person.

        Returns:
            List[int]: A binary vector indicating the topics (clusters) the person belongs to.
        """
        if not self._is_trained:
            raise Exception("Train must be called before generating results.")

        # Predict the clusters for each embedding
        cluster_assignments = self._kmeans.predict(person_embeddings)

        # Generate a binary vector indicating topic membership
        result_vector = [0] * self.topic_count
        for cluster in cluster_assignments:
            result_vector[cluster] = 1

        return result_vector
    
    def topic_map(self, embedding: list[float]) -> int:
        """
        Returns topic corresponding to an embedding
        """
        if not self._is_trained:
            raise Exception("Train must be called before generating results.")
        
        return int(self._kmeans.predict([embedding])[0])


    def is_finished_training(self) -> bool:
        """
        Returns whether the training is complete.

        Returns:
            bool: True if the model is trained, False otherwise.
        """
        return self._is_trained


    def getStats(self, batch_size: int = 20000):
        """
        Computes clustering statistics efficiently using a single batch of embeddings.

        Args:
            batch_size (int): The number of embeddings to sample for computing metrics.

        Returns:
            dict: A dictionary with clustering statistics.
        """
        if not self._is_trained:
            raise Exception("Train must be called before computing stats.")

        # Get a single batch of embeddings
        batch_size = 20000
        epoch = random.randint(0, 100)
        # Get embeddings and corresponding topic assignments
        _, sample_embeddings = self._embeddings.getEmbeddingBatch(0, batch_size, epoch)
        # Ensure we have enough samples
        if len(sample_embeddings) < 2:
            return {
                "Inertia (Compactness)": self._kmeans.inertia_,
                "Silhouette Score (Separation + Compactness)": -1,
                "Davies-Bouldin Index (Separation)": -1,
                "Calinski-Harabasz Score (Compactness vs Separation)": -1,
                "Cluster Counts": {i: 0 for i in range(self.topic_count)}
            }

        # Predict clusters for the sampled embeddings
        sample_labels = self._kmeans.predict(sample_embeddings)

        # Compute cluster statistics
        silhouette = silhouette_score(sample_embeddings, sample_labels)
        davies_bouldin = davies_bouldin_score(sample_embeddings, sample_labels)
        calinski_harabasz = calinski_harabasz_score(sample_embeddings, sample_labels)

        # Compute cluster counts
        cluster_counts = {i: int(np.sum(sample_labels == i)) for i in range(self.topic_count)}

        # Return the stats
        stats = {
            "Inertia (Compactness)": self._kmeans.inertia_,
            "Silhouette Score (Separation + Compactness)": silhouette,
            "Davies-Bouldin Index (Separation)": davies_bouldin,
            "Calinski-Harabasz Score (Compactness vs Separation)": calinski_harabasz,
            "Cluster Counts": cluster_counts
        }

        return stats

