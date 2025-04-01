import numpy as np
import umap
import matplotlib.pyplot as plt
from ..helpers.embeddings import Embeddings
from ..helpers.prototypes import ClusteringAgent
import random 
import time
import IPython.display as display


class Umap:
    def __init__(self, embeddings, clustering_agent, dims = 2, umap_model=None): 
        self.embeddings = embeddings
        self.clustering_agent = clustering_agent
        self.topic_count = clustering_agent.topic_count
        self.umap_model = umap_model
        self.dims = dims

        embeddings, self.topics = self.__fetch_embeddings(self.embeddings, self.clustering_agent)

        # Reduce dimensionality using UMAP
        if self.umap_model is None: 
            reducer = umap.UMAP(n_components=self.dims)
        else:
            reducer = umap_model
        self.reduced_embeddings = reducer.fit_transform(embeddings)
        self.ax = None
        self.fig = None

        print('✅ Umap initialized!')

    def resetAxes(self):
        if (self.ax): 
            self.ax.clear()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

    # Load embeddings and their topic assignments from your DB
    def __fetch_embeddings(self, embeddings: Embeddings, clustering_agent: ClusteringAgent):
        if not clustering_agent.is_finished_training():
            raise Exception("Clustering Agent must be trained before running tSNE")
        
        ## Get Embeddings
        batch_size = 20000
        epoch = random.randint(0, 100)
        # Get embeddings and corresponding topic assignments
        _, embeddings = embeddings.getEmbeddingBatch(0, batch_size, epoch)
        
        topics = []
        for e in embeddings: 
            topics.append(clustering_agent.topic_map(e))


        return embeddings, topics

    # Fetch data
    def colorMap(self): 
        self.resetAxes()

        if self.dims == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')  # Set to 3D
            scatter = self.ax.scatter(self.reduced_embeddings[:, 0], 
                                    self.reduced_embeddings[:, 1], 
                                    self.reduced_embeddings[:, 2], 
                                    c=self.topics, cmap="tab10", alpha=0.5, edgecolors='k', linewidth=0.2)
            
            self.ax.set_zlabel("Projection Axis 3")  # Add Z-axis label
        else:
            self.ax = self.fig.add_subplot(111)  # Set to 2D
            scatter = self.ax.scatter(self.reduced_embeddings[:, 0], 
                                    self.reduced_embeddings[:, 1], 
                                    c=self.topics, cmap="tab10", alpha=0.5, edgecolors='k', linewidth=0.2)

        self.ax.set_title("Topic Space Visualization (UMAP Projection)")
        self.ax.set_xlabel("Projection Axis 1")
        self.ax.set_ylabel("Projection Axis 2")
        self.ax.grid(True, linestyle="--", alpha=0.3)

        # Update colorbar
        self.fig.colorbar(scatter, ax=self.ax, label="Topic Cluster ID")

        self.fig.canvas.draw_idle()  # Update the existing figure


    def highlightTopic(self, focus_topic: int ): 
        if focus_topic < 0 or focus_topic >= self.topic_count:
            raise Exception(f"Selected topic must be in range [0, {self.topic_count})")
        
        self.resetAxes()
    
        mask = np.array(self.topics) == focus_topic
        self.ax.scatter(self.reduced_embeddings[:, 0], self.reduced_embeddings[:, 1], color='gray', alpha=0.3)
        self.ax.scatter(self.reduced_embeddings[mask, 0], self.reduced_embeddings[mask, 1], 
                        color='red', label=f'Topic {focus_topic}')
        
        self.ax.set_title(f"Highlighting Topic {focus_topic} [{len(self.reduced_embeddings[mask, 0])} points]")
        self.ax.legend()    

        self.fig.canvas.draw_idle()
        display.display(self.fig)
        display.clear_output(wait=True)  # Remove flickering by clearing the output before showing the new one


    def rotateTopics(self): 
        while True: 
            for i in range(self.topic_count):
                self.highlightTopic(i)
                time.sleep(0.5)

    def getClusterMembership(self):
        return [int(np.sum(np.array(self.topics) == i-1 )) for i in range(self.topic_count)]
