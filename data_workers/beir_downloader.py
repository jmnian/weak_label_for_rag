from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset

'''
Go to https://github.com/beir-cellar/beir?tab=readme-ov-file#beers-available-datasets
'''

dataset = "nfcorpus"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = "/WAVE/datasets/yfang_lab/QA_datasets"
data_path = util.download_and_unzip(url, out_dir)

# #### Provide the data_path where scifact has been downloaded and unzipped
# corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# #### Load the SBERT model and retrieve using cosine-similarity
# model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16)
# retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
# results = retriever.retrieve(corpus, queries)

# #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
# ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)