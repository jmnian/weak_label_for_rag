from sentence_transformers import SentenceTransformer
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.train import TrainRetriever
from time import time
import argparse, logging, json, os
os.environ['HF_HOME'] = '/local/scratch'


parser = argparse.ArgumentParser(description="Script for training a retriever")
parser.add_argument('--data_path', type=str, help='Path to the corpus, queries, etc.')
parser.add_argument('--model_path', type=str, help='Path to the model that you want to evaluate')
parser.add_argument('--eval_recontriever', action='store_true', help="y for evaluating recontriever")
parser.add_argument('--eval_contriever', action='store_true', help="y for evaluating recontriever")
args = parser.parse_args()

split = "test"
if args.eval_recontriever:
    model_name = "Yibin-Lei/ReContriever"
    print(f"Evaluating ReContriever on the {split}.tsv file in {args.data_path.split('/')[-1]}")
elif args.eval_contriever:
    model_name = "facebook/contriever" 
    print(f"Evaluating Contriever on the {split}.tsv file in {args.data_path.split('/')[-1]}")
else:
    print(f"Evaluating {args.model_path.split('/')[-1]} on the {split}.tsv file in {args.data_path.split('/')[-1]}")

#### Load Data
corpus, queries, qrels = GenericDataLoader(args.data_path).load(split=split)


#### Load model and then evaluate 
if args.eval_recontriever or args.eval_contriever:
    model = DRES(models.SentenceBERT(model_name), batch_size=256, corpus_chunk_size=1_000_000)
    retriever = EvaluateRetrieval(model, k_values=[1,3,5,10,100], score_function="cos")

    #### Retrieve dense results (format of results is identical to qrels)
    start_time = time()
    results = retriever.retrieve(corpus, queries)
    end_time = time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

    print(recall)
    print(mrr)

    
else: # my trained dprs 
    model = SentenceTransformer(args.model_path) 

    retriever = TrainRetriever(model=model, batch_size=16)

    ir_evaluator = retriever.load_ir_evaluator(corpus, queries, qrels)

    scores = ir_evaluator.compute_metrices(model)

    print(json.dumps(scores, indent=4))