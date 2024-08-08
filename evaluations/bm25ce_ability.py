from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank, models 
from time import time
import argparse, logging, json, os
from bm25_evaluation import load_or_create_bm25
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
#### Load Model
cross_encoder_model = CrossEncoder(model_name)
reranker = Rerank(cross_encoder_model, batch_size=128)

#### Format results into {qid: {pid: score, pid:score, ...}}
results = {}
bm25 = load_or_create_bm25(corpus, f"{data_path}/bm25_{len(corpus)}.pkl")

ranked_lists = {}
for query_id, corpus_id in tqdm(qrels.items(), desc="Processing Queries"):
    query = queries[query_id]
    scores = bm25.get_scores(query)
    ranked_indices = np.argsort(scores)[::-1][:args.bm25_topk]
    ranked_docs = [list(corpus.keys())[idx] for idx in ranked_indices]
    ranked_lists[query_id] = ranked_docs


results = reranker.rerank(corpus, queries, results, top_k=args.bm25_topk) 