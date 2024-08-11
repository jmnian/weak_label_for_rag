from sentence_transformers import SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import List, Dict

import logging, argparse 
import numpy as np
import pathlib, os
import random

class CustomModel:
    def __init__(self, model_path=None, **kwargs):
        self.model = SentenceTransformer(model_path)
    
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    # For eg ==> return np.asarray(self.model.encode(queries, batch_size=batch_size, **kwargs))
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        tensor = self.model.encode(queries, batch_size=batch_size, **kwargs)
        return tensor.cpu().numpy()
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    # For eg ==> sentences = [(doc["title"] + "  " + doc["text"]).strip() for doc in corpus]
    #        ==> return np.asarray(self.model.encode(sentences, batch_size=batch_size, **kwargs))
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + "  " + doc["text"]).strip() for doc in corpus]
        tensor = self.model.encode(sentences, batch_size=batch_size, **kwargs)
        return tensor.cpu().numpy()


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser(description="Script for training a retriever")
parser.add_argument('--beir_dataset_path', type=str, help='Path to the BEIR different dataset\'s corpus, queries, etc.')
parser.add_argument('--model_path', type=str, help='Which retriever you want to evaluate')
parser.add_argument('--dataset_name', type=str, help='Which dataset you want to evaluate on')
args = parser.parse_args()


#### Download nfcorpus.zip dataset and unzip the dataset
dataset = args.dataset_name
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = args.beir_dataset_path
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Provide your custom model class name --> HERE
model = DRES(CustomModel(model_path=args.model_path))

retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "dot" if you wish dot-product

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
