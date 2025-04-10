# Evaluating Domain-Specific Topic Reduction for Sparse Vector Document Retrieval

## Abstract

This thesis investigates the limitations of current document retrieval systems and introduces an alternative architecture leveraging topic-level sparse indexing of contextual embeddings. This theoretical retrieval system seeks to achieve high computational efficiency through low latency and indexing overhead, while also achieving high semantic understanding and respecting local meaning and document cohesion. Additionally, the system supports scalable and context-aware document matching without reliance on user interaction data

In pursuit of these objectives, the system makes 2 key assumptions on the structure and content of documents within a chosen application domain. The first assumption is that documents can be broken into self-contained semantic components, the second assumes an ability to represent the application domain's distinct meanings as a finite, discrete set of topics. 

At a high level, the proposed system aims to represent a document as a bag of topics, then apply sparse vector ranking algorithms at retrieval time. Topics are inferred by clustering the contextualized embeddings of semantic components within a learned embedding space.

The contributions of this thesis involve a review of existing retrieval methods, an outline of the proposed system's intuition and architecture, and an explorative implementation against a strategically chosen application domain. The thesis finds that standard embedding models (SBERT in this case) are insufficient for identifying application specific topics. Future work will focus on fine-tuning embedding models to better capture domain-specific semantics and fully evaluate the potential of this topic-based retrieval framework.

The thesis also provides the necessary tooling, for extension and modification of the retrieval pipeline. Namely, it supports the training and querying of the proposed retrieval system, while accepting custom implementations at each step. 

## ResearchRunner
To run an experiment:
1) Clone the repository to your local device
2) Unzip datasets.zip and notebooks.zip
3) Set up a virtual environment and install requirements.txt
4) Use notebooks/datasetGenerator to ingest the data
5) Implement ChunkingAgent, EmbeddingAgent, and ClusteringAgent Prototypes (see below)
6) Initialize a ResearchRunner instance using these agents (see below)
7) Call the runResearch() method on ResearchRunner
8) Once trained, call query() on ResearchRunner to get the document ids of top results

## Agent Prototypes
```python
## Chunking agents implement segmentation step
class ChunkingAgent(Protocol):
    name: str

    ## Extract semantic components from text
    def chunk(self, raw_text: str) -> list[str]:

## Embedding agents implement the representation step
class EmbeddingAgent(Protocol):
    name: str

    ## Generate a text embedding for a semantic component
    def embed(self, raw_text: list[str]) -> list[list[float]]:
        ...

## Clustering agents implement the topic reduction step
class ClusteringAgent(Protocol):
    name: str
    
    ## Pass the embeddings object (used by pipeline to access embeddings table
    def pass_embeddings(self, embeddings: Embeddings):
        ...
    ## Learn topics, pass_embeddings must have been called previously
    def train(embeddings: list[list[float]]):
        ...
    ## Get sparse vector representation for component embeddings 
    def generate_result(self, person_embeddings: list[list[float]]) -> list[int]:
        ...
    # Classify an embedding into a topic
    def topic_map(embedding: list[float]) -> int:
        ...
    ## True if train has been called previously 
    def is_finished_training() -> bool:
        ...
```

## Sample Usage 
```python
sentenceChunker = SentenceChunkingAgent()
sbert = SBERTEmbeddingAgent(device=device)
kmeans = KMeansClusteringAgent(200)

rr = ResearchRunner('test_db', 'sentence_sbert_kmeans_100', sentenceChunker, sbert, kmeans)
```

[!ResearchRunner Screenshot](https://github.com/cirons2003/document_retrieval_research/blob/master/unnamed.png)



## Full Thesis 
[[![Download Thesis](https://img.shields.io/badge/View-PDF-blue)](https://github.com/cirons2003/document_retrieval_research/blob/master/thesis.pdf)]





## Authors
Carson Irons  
LinkedIn: linkedin.com/in/carson-irons-9ab55a23b
