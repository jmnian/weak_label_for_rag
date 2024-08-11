from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank, models 
from time import time
import argparse, logging, json, os
import numpy as np
import eval_util
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from nltk.tokenize import word_tokenize
os.environ['HF_HOME'] = '/local/scratch'


parser = argparse.ArgumentParser(description="Script for training a retriever")
parser.add_argument('--data_path', type=str, help='Path to the corpus, queries, etc.')
parser.add_argument('--model_path', type=str, help='Path to the model that you want to evaluate')
parser.add_argument('--eval_recontriever', action='store_true', help="y for evaluating recontriever")
parser.add_argument('--eval_contriever', action='store_true', help="y for evaluating recontriever")
parser.add_argument('--bm25_topk', type=int, help="How many documents BM25 retrieves for each question")
args = parser.parse_args()


split = "test"
if args.eval_recontriever:
    model_name = "Yibin-Lei/ReContriever"
    print(f"Evaluating BM25+CE(ReContriever) on the {split}.tsv file in {args.data_path.split('/')[-1]}")
elif args.eval_contriever:
    model_name = "facebook/contriever" 
    print(f"Evaluating BM25+CE(Contriever) on the {split}.tsv file in {args.data_path.split('/')[-1]}")
else:
    model_name = args.model_path
    print(f"Evaluating BM25+CE({args.model_path.split('/')[-1]}) on the {split}.tsv file in {args.data_path.split('/')[-1]}")

#### Load Data
corpus, queries, qrels = GenericDataLoader(args.data_path).load(split=split)

def load_or_create_bm25_results(data_path, corpus, qrels, queries, bm25_topk=100):
    results_file = os.path.join(data_path, f'bm25_top{bm25_topk}_on_test.json')

    if os.path.isfile(results_file):
        print("Load existing results from the JSON file", results_file)
        with open(results_file, 'r') as f:
            results = json.load(f)
    else: 
        print("No Results file found, creating one right now. ")
        results = {}
        
        bm25 = eval_util.load_or_create_bm25(corpus, f"{data_path}/bm25_{len(corpus)}.pkl")
        
        for query_id, relevant_docs in tqdm(qrels.items(), desc="BM25 retrieving"):
            query = queries[query_id]
            query_tokens = word_tokenize(query.lower())
            scores = bm25.get_scores(query_tokens)
            ranked_indices = np.argsort(scores)[::-1][:bm25_topk]
            results[query_id] = {str(idx): scores[idx] for idx in ranked_indices}

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print("BM25 Results saved at", results_file)

    return results

#### Format results into {qid: {pid: score, pid:score, ...}}
bm25_results = load_or_create_bm25_results(args.data_path, corpus, qrels, queries, args.bm25_topk)

#### Load Model
cross_encoder_model = CrossEncoder(model_name)
reranker = Rerank(cross_encoder_model, batch_size=128)
start_time = time()
rerank_results = reranker.rerank(corpus, queries, bm25_results, top_k=args.bm25_topk) 
end_time = time()
print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

retriever = EvaluateRetrieval(k_values=[1,3,5,10,100], score_function="cos_sim")
bm25_ndcg, bm25_map, bm25_recall, _ = EvaluateRetrieval.evaluate(qrels, bm25_results, retriever.k_values)
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)
bm25_mrr = EvaluateRetrieval.evaluate_custom(qrels, bm25_results, retriever.k_values, metric="mrr")
mrr = EvaluateRetrieval.evaluate_custom(qrels, rerank_results, retriever.k_values, metric="mrr")

print(f"BM25 results: ")
print(bm25_ndcg)
print(bm25_map)
print(bm25_recall)
print(bm25_mrr)

print(f"BM25 retrieve top{args.bm25_topk} then CE rerank results: ({args.model_path.split('/')[-1]})")
print(ndcg)
print(_map)
print(recall)
print(mrr)