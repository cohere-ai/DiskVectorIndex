# DiskVectorIndex - Ultra-Low Memory Vector Search on Large Dataset

Indexing large datasets (100M+ embeddings) requires a lot of memory in most vector databases: For 100M documents/embeddings, most vector databases require about **500GB of memory**, driving the cost for your servers accordingly high.

This repository offers methods to be able to search on very large datasets (100M+) with just **300MB of memory**, making semantic search on such large datasets suitable for the Memory-Poor developers.

We provide various pre-build indices, that can be used to semantic search and powering your RAG applications.

## Pre-Build Indices

Below you find different pre-build indices. The embeddings are downloaded at the first call, the size is specified under Index Size. Most of the embeddings are memory mapped from disk, e.g. for the `Cohere/trec-rag-2024-index` corpus you need 15 GB of disk, but just 380 MB of memory to load the index.

| Name | Description | #Docs | Index Size (GB) | Memory Needed |
| --- | --- | :---: | :---: | :---: | 
|  [Cohere/trec-rag-2024-index](https://huggingface.co/datasets/Cohere/trec-rag-2024-index) | Segmented corpus for [TREC RAG 2024](https://trec-rag.github.io/annoucements/2024-corpus-finalization/) | 113,520,750 | 15GB | 380MB |
| fineweb-edu-10B-index (soon)  | 10B token sample from [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) embedded and indexed on document level. | 9,267,429 | 1.4GB | 230MB |
| fineweb-edu-100B-index (soon)  | 100B token sample from [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) embedded and indexed on document level. | 69,672,066 | 9.2GB | 380MB
| fineweb-edu-350B-index (soon)  | 350B token sample from [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) embedded and indexed on document level. | 160,198,578 | 21GB | 380MB
| fineweb-edu-index (soon) | Full 1.3T token dataset [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) embedded and indexed on document level. | 324,322,256 | 42GB | 285MB


Each index comes with the respective corpus, that is chunked into smaller parts. These chunks are downloaded on-demand and reused for further queries.

## Getting Started

Get your free **Cohere API key** from [cohere.com](https://cohere.com). You must set this API key as an environment variable: 
```
export COHERE_API_KEY=your_api_key
```


Make sure you have `wget` installed, to be able to download files on-the-fly from the HuggingFace hub.

Install the package:
```
pip install DiskVectorIndex
```

You can then search via:
```python
from DiskVectorIndex import DiskVectorIndex

index = DiskVectorIndex("Cohere/trec-rag-2024-index")

while True:
    query = input("\n\nEnter a question: ")
    docs = index.search(query, top_k=3)
    for doc in docs:
        print(doc)
        print("=========")
```


You can also load a fully downloaded index from disk via:
```python
from DiskVectorIndex import DiskVectorIndex

index = DiskVectorIndex("path/to/index")
```


# How does it work?
The Cohere embeddings have been optimized to work well in compressed vector space, as detailed in our [Cohere int8 & binary Embeddings blog post](https://cohere.com/blog/int8-binary-embeddings). The embeddings have not only been trained to work in float32, which requires a lot of memory, but to also operate well with int8, binary and Product Quantization (PQ) compression.

The above indices uses Product Quantization (PQ) to go from originally 1024*4=4096 bytes per embedding to just 128 bytes per embedding, reducing your memory requirement 32x.

Further, we use [faiss](https://github.com/facebookresearch/faiss) with a memory mapped IVF: In this case, only a small fraction (between 32,768 and 131,072) embeddings must be loaded in memory. 


# Need Semantic Search at Scale?

At [Cohere](https://cohere.com) we helped customers to run Semantic Search on tens of billions of embeddings, at a fraction of the cost. Feel free to reach out for [Nils Reimers](mailto:nils@cohere.com) if you need a solution that scales.