from sklearn.decomposition import IncrementalPCA
from umap import UMAP
import hdbscan
import numpy as np

def countTopics(embeddings) -> int:
    batch_size = 5000  # Adjust for memory efficiency
    n_components_pca = 100  # Reduce to 100 dimensions before UMAP

    # Step 1: Incremental PCA (Batch Processing)
    pca = IncrementalPCA(n_components=n_components_pca)
    
    index = 0
    
    while True: 
        batch = embeddings.getEmbeddingBatchRaw(index, batch_size)
        if not batch: 
            break 
        pca.partial_fit(batch)  # Fit PCA incrementally
        index += len(batch)

    reduced_embeddings = []  # Store PCA-reduced embeddings

    while True: 
        batch = embeddings.getEmbeddingBatchRaw(index, batch_size)
        if not batch: 
            break 
        reduced_batch = pca.transform(batch)  # Reduce with PCA
        reduced_embeddings.append(reduced_batch)
        index += len(batch)

    reduced_embeddings = np.vstack(reduced_embeddings)  # Merge batches

    print('made it!')

    # Step 2: Apply UMAP once on the fully reduced dataset
    reducer = UMAP(n_components=50, n_neighbors=15, min_dist=0.1)
    umap_embeddings = reducer.fit_transform(reduced_embeddings)

    # Step 3: Cluster using HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=100,  
        min_samples=10,  
        cluster_selection_method='eom',  
        approx_min_span_tree=True  
    )
    cluster_labels = clusterer.fit_predict(umap_embeddings)

    # Return the number of clusters (excluding noise points, -1 labels)
    return len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
