import os
import pickle
import jsonlines
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from collections import defaultdict

def load_or_create_bm25(corpus, filename):
    # Check if the serialized BM25 object exists
    if os.path.exists(filename):
        print(f"Loading BM25 object from {filename}")
        with open(filename, 'rb') as f:
            bm25 = pickle.load(f)
        print(f"BM25 loaded successfully")
    else:
        print("BM25 object not found, creating now...")
        tokenized_corpus = {pid: word_tokenize(text.lower()) for pid, text in corpus.items()}
        sorted_tokenized_passages = [tokenized_corpus[pid] for pid in sorted(tokenized_corpus)]
        bm25 = BM25Okapi(sorted_tokenized_passages)
        with open(filename, 'wb') as f:
            pickle.dump(bm25, f)
        print(f"BM25 object created and saved to {filename}")
    return bm25

def load_corpus(corpus_path):
    corpus = {}
    with jsonlines.open(corpus_path) as reader:
        for obj in reader:
            corpus[obj["_id"]] = obj["text"]
    return corpus

def load_queries(queries_path):
    queries = {}
    with jsonlines.open(queries_path) as reader:
        for obj in reader:
            queries[obj["_id"]] = word_tokenize(obj["text"].lower())
    return queries

def load_qrel(qrel_path):
    qrel = pd.read_csv(qrel_path, sep='\t', header=None, names=["query-id", "corpus-id", "score"])
    return qrel

class BM25WithProgress(BM25Okapi):
    def __init__(self, corpus, progress_bar=True):
        if progress_bar:
            corpus = list(tqdm(corpus, desc="Indexing Corpus", unit="document"))
        super().__init__(corpus)

def compute_metrics(ranked_lists, qrel_dict, ks=[1, 3, 5, 10, 50, 120]):
    recall_at_k = {k: [] for k in ks}
    mrr_at_5 = []
    hit_rate_at_5 = []
    
    for query_id, ranked_docs in ranked_lists.items():
        relevant_docs = qrel_dict.get(query_id, [])
        if not relevant_docs:
            continue
        hits = np.isin(ranked_docs, relevant_docs)
        
        for k in ks:
            recall_at_k[k].append(hits[:k].sum() / len(relevant_docs))
        
        mrr = 0
        hit = 0
        for rank, doc in enumerate(ranked_docs[:5], start=1):
            if doc in relevant_docs:
                mrr = 1 / rank
                hit = 1
                break
        mrr_at_5.append(mrr)
        hit_rate_at_5.append(hit)
    
    recall_at_k = {k: np.mean(v) for k, v in recall_at_k.items()}
    mean_mrr_at_5 = np.mean(mrr_at_5)
    mean_hit_rate_at_5 = np.mean(hit_rate_at_5)
    
    return recall_at_k, mean_mrr_at_5, mean_hit_rate_at_5

def bm25_evaluation(corpus_path, queries_path, qrel_path, retrieve_top_k, bm25_pkl_path):
    # Load data
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrel = load_qrel(qrel_path)
    
    # Create BM25 object
    bm25 = load_or_create_bm25(corpus, bm25_pkl_path)
    
    # Create qrel_dict for easy lookup
    qrel_dict = defaultdict(list)
    for _, row in qrel.iterrows():
        if row["score"] == '1':
            qrel_dict[row["query-id"]].append(str(row["corpus-id"]))  # Ensure corpus-id is a string
    # Perform search and collect results
    ranked_lists = {}
    for query_id, corpus_id in tqdm(qrel_dict.items(), desc="Processing Queries"):
        query = queries[query_id]
        query_tokens = word_tokenize(query.lower())
        scores = bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:retrieve_top_k]
        ranked_docs = [list(corpus.keys())[idx] for idx in ranked_indices]
        ranked_lists[query_id] = ranked_docs
    
    # Compute metrics
    recall_at_k, mean_mrr_at_5, mean_hit_rate_at_5 = compute_metrics(ranked_lists, qrel_dict)
    
    # Print results
    for k, recall in recall_at_k.items():
        print(f"Recall@{k}: {recall:.4f}")
    print(f"MRR@5: {mean_mrr_at_5:.4f}")
    print(f"Hit Rate@5: {mean_hit_rate_at_5:.4f}")


# Example usage
corpus_path    = "/WAVE/users2/unix/jnian/WeakLabelForRAG/data/wq_train_corpus163683_weakTrainQ2000_ValQ474/corpus.jsonl"
queries_path   = "/WAVE/users2/unix/jnian/WeakLabelForRAG/data/wq_train_corpus163683_weakTrainQ2000_ValQ474/queries.jsonl"
test_qrel_path = "/WAVE/users2/unix/jnian/WeakLabelForRAG/data/wq_train_corpus163683_weakTrainQ2000_ValQ474/qrels/val_groundtruth.tsv"
bm25_pkl_path  = "/WAVE/users2/unix/jnian/WeakLabelForRAG/data/wq_train_corpus163683_weakTrainQ2000_ValQ474/bm25_163683.pkl"
retrieve_top_k = 120
bm25_evaluation(corpus_path, queries_path, test_qrel_path, retrieve_top_k, bm25_pkl_path)