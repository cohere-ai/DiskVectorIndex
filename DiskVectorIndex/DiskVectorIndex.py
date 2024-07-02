import faiss
import cohere 
import os
import numpy as np
import json
from indexed_zstd import IndexedZstdFile
import time 
import logging
import psutil
import sys
import requests
import tqdm

process = psutil.Process()
faiss.omp_set_num_threads(1)
logger = logging.getLogger(__name__)

class DiskVectorIndex:
    def __init__(self, index_name_or_path, cache_dir="index_cache", nprobe=None):
        if 'COHERE_API_KEY' not in os.environ:
            raise Exception("Please set the COHERE_API_KEY environment variable to your Cohere API key.")
        
        self.co = cohere.Client(os.environ['COHERE_API_KEY']) 
        self.remote_path = None 

        if os.path.exists(index_name_or_path):
            self.local_dir = index_name_or_path
        else:
            self.local_dir = os.path.join(cache_dir, index_name_or_path.replace("/", "_"))
            os.makedirs(self.local_dir, exist_ok=True)
            self.remote_path = index_name_or_path

        #Load the config
        config_path = os.path.join(self.local_dir, "config.json")
        self.download_from_remote("config.json")

        if not os.path.exists(config_path):
            raise Exception(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as fIn:
            self.config = json.load(fIn)

        #Load the index
        self.download_from_remote(self.config["index"])
        index_path = os.path.join(self.local_dir, self.config["index"])
        start_time = time.time()
        self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        if nprobe is not None:
            self.index.nprobe = nprobe
       
        logger.info(f"Index load time: {(time.time()-start_time)*1000:.2f} ms, {process.memory_info().rss / 1024 / 1024:.2f} MB")
        logger.info(f"Index loaded with {self.index.ntotal} vectors, {self.index.nprobe} nprobe, {self.index.nlist} nlist")

    def search(self, query, top_k=10):
        embedding_type = self.config.get('embedding_type', 'float')
        embeddings = self.co.embed(texts=[query], model=self.config['model'], input_type="search_query", embedding_types=[embedding_type]).embeddings
        embeddings = getattr(embeddings, embedding_type)
        query_emb = np.asarray(embeddings)

        start_time = time.time()
        scores, doc_indices = self.index.search(query_emb, top_k)
        logging.info(f"Search time: {(time.time()-start_time)*1000:.2f} ms")

        scores = scores[0].tolist()
        doc_indices = doc_indices[0].tolist()

        docs = []
        start_time = time.time()
        for score, doc_idx in zip(scores, doc_indices):
            corpus_file_id = doc_idx // self.config['corpus_num_lines']
            corpus_file_id_str = str(corpus_file_id).zfill(self.config['corpus_file_len'])

            corpus_file_path = os.path.join("corpus", corpus_file_id_str[-self.config['corpus_folder_len']:], f"{corpus_file_id_str}.jsonl.zst")
            corpus_offset_path = os.path.join("corpus", corpus_file_id_str[-self.config['corpus_folder_len']:], f"{corpus_file_id_str}.jsonl.offsets")
            self.download_from_remote(corpus_file_path)
            self.download_from_remote(corpus_offset_path)

            offsets = np.load(os.path.join(self.local_dir, corpus_offset_path), mmap_mode="r")
            with IndexedZstdFile(os.path.join(self.local_dir, corpus_file_path)) as fCorpus:
                fCorpus.seek(offsets[doc_idx % self.config['corpus_num_lines']])
                doc = json.loads(fCorpus.readline())
                docs.append({'doc': doc, 'score': score})

        logging.info(f"Document fetch time: {(time.time()-start_time)*1000:.2f} ms")
        return docs 

    def download_from_remote(self, filename):
        if self.remote_path is None:
            return
        
        local_filepath = os.path.join(self.local_dir, filename)

        if os.path.exists(local_filepath):
            return
        
        url = f"https://huggingface.co/datasets/{self.remote_path}/resolve/main/{filename}"
        os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
        print(f"Downloading file: {url}")
        self.http_get(url, local_filepath)


    def http_get(self, url, path):
        """
        Downloads a URL to a given path on disc
        """
        if os.path.dirname(path) != '':
            os.makedirs(os.path.dirname(path), exist_ok=True)

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
            req.raise_for_status()
            return

        download_filepath = path+"_part"
        with open(download_filepath, "wb") as file_binary:
            content_length = req.headers.get('Content-Length')
            total = int(content_length) if content_length is not None else None
            progress = tqdm.tqdm(unit="B", total=total, unit_scale=True)
            for chunk in req.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    file_binary.write(chunk)

        os.rename(download_filepath, path)
        progress.close()