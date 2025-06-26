# DiskVectorIndex - Ultra-Low Memory Vector Search on Large Dataset

Indexing large datasets (100M+ embeddings) requires a lot of memory in most vector databases: For 100M documents/embeddings, most vector databases require about **500GB of memory**, driving the cost for your servers accordingly high.

This repository offers methods to be able to search on very large datasets (100M+) with just **300MB of memory**, making semantic search on such large datasets suitable for the Memory-Poor developers.

We provide various pre-build indices, that can be used to semantic search and powering your RAG applications.

## Pre-Build Indices

Below you find different pre-build indices. The embeddings are downloaded at the first call, the size is specified under Index Size. Most of the embeddings are memory mapped from disk, e.g. for the `Cohere/trec-rag-2024-index` corpus you need 15 GB of disk, but just 380 MB of memory to load the index.

| Name | Description | #Docs | Index Size (GB) | Memory Needed |
| --- | --- | :---: | :---: | :---: | 
|  [Cohere/trec-rag-2024-index](https://huggingface.co/datasets/Cohere/trec-rag-2024-index) | Segmented corpus for [TREC RAG 2024](https://trec-rag.github.io/annoucements/2024-corpus-finalization/) | 113,520,750 | 15GB | 380MB |
| [Cohere/fineweb-edu-10B-index](https://huggingface.co/datasets/Cohere/fineweb-edu-10B-index)  | 10B token sample from [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) embedded and indexed on document level. | 9,267,429 | 1.4GB | 230MB |
| [Cohere/fineweb-edu-100B-index](https://huggingface.co/datasets/Cohere/fineweb-edu-100B-index)  | 100B token sample from [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) embedded and indexed on document level. | 69,672,066 | 9.2GB | 380MB
| [Cohere/fineweb-edu-350B-index](https://huggingface.co/datasets/Cohere/fineweb-edu-350B-index)  | 350B token sample from [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) embedded and indexed on document level. | 160,198,578 | 21GB | 380MB
| [Cohere/fineweb-edu-index](https://huggingface.co/datasets/Cohere/fineweb-edu-index) | Full 1.3T token dataset [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) embedded and indexed on document level. | 324,322,256 | 42GB | 285MB


Each index comes with the respective corpus, that is chunked into smaller parts. These chunks are downloaded on-demand and reused for further queries.

## Getting Started

Get your free **Cohere API key** from [cohere.com](https://cohere.com). You must set this API key as an environment variable: 
```
export COHERE_API_KEY=your_api_key
```

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

# End2End RAG Example

We can use the excellent RAG capabilities of the [Cohere Command R+](https://docs.cohere.com/docs/retrieval-augmented-generation-rag) model to build an end2end RAG pipeline:

```python
import cohere
from DiskVectorIndex import DiskVectorIndex
import os 
import sys 

co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
index = DiskVectorIndex("Cohere/trec-rag-2024-index")

question = "Which popular deep learning frameworks were developed by Facebook and Google? What are their differences?"
prompt = f"Answer the following question with a detailed answer: {question}"


print("Question:", question)

# Step 1 - Decompose the question into sub-questions
res = co.chat(
  model="command-r-plus",
  message=prompt,
  search_queries_only=True
)

sub_queries = [r.text for r in res.search_queries]
print("Generated sub queries:", sub_queries)

# Step 2 - Search for relevant documents for each sub 
print("Start searching")
docs = []
doc_id = 1
for query in sub_queries:
    hits = index.search(query, top_k=3)
    for hit in hits:
        docs.append({"id": str(doc_id), 'title': hit['doc']['title'], 'snippet': hit['doc']['segment']})
        doc_id += 1

print(f"Documents found: {len(docs)}")

# Step 3 - Generate the response
print("Start generating response")
print("==============")

for event in co.chat_stream(model="command-r-plus", message=prompt, documents=docs, citation_quality="fast"):
    if event.event_type == "text-generation":
        #Print a text chunk
        print(event.text, end="")
    elif event.event_type == "citation-generation":
        #Print the citations as inline citations
        print("["+", ".join(event.citations[0].document_ids)+"]", end="")
```

# How does the DiskVectorIndex work?
The Cohere embeddings have been optimized to work well in compressed vector space, as detailed in our [Cohere int8 & binary Embeddings blog post](https://cohere.com/blog/int8-binary-embeddings). The embeddings have not only been trained to work in float32, which requires a lot of memory, but to also operate well with int8, binary and Product Quantization (PQ) compression.

The above indices uses Product Quantization (PQ) to go from originally 1024*4=4096 bytes per embedding to just 128 bytes per embedding, reducing your memory requirement 32x.

Further, we use [faiss](https://github.com/facebookresearch/faiss) with a memory mapped IVF: In this case, only a small fraction (between 32,768 and 131,072) embeddings must be loaded in memory. 


# Need Semantic Search at Scale?

At [Cohere](https://cohere.com) we helped customers to run Semantic Search on tens of billions of embeddings, at a fraction of the cost. Feel free to reach out for [Nils Reimers](mailto:nils@cohere.com) if you need a solution that scales.
