from ragatouille import RAGPretrainedModel
from beir.datasets.data_loader import GenericDataLoader
import argparse, logging, json, os

parser = argparse.ArgumentParser(description="Script for training a retriever")
parser.add_argument('--data_path', type=str, help='Path to the corpus, queries, etc.')
parser.add_argument('--model_path', type=str, help='Path to the model that you want to evaluate')
parser.add_argument('--index_name', type=str, help="Give the index a name, or try to load an existing index")
args = parser.parse_args()

path_to_index = f".ragatouille/colbert/indexes/{args.index_name}"
if os.path.exists(path_to_index):
    print(f"Loading index: {args.index_name}")
    RAG = RAGPretrainedModel.from_index(path_to_index)
else:
    RAG = RAGPretrainedModel.from_pretrained(args.model_path)
    corpus, queries, qrels = GenericDataLoader(args.data_path).load(split="test")
    all_documents = [f"{item['text']} {item['title']}" for item in corpus.values()]
    print(f"Building index: {args.index_name}")
    RAG.index(
        collection=all_documents, 
        index_name=args.index_name, 
        max_document_length=350, 
        use_faiss=True
        )

k = 3 
# ground truth passage should be: Cross-Country Skiing Burns More. Burning about 381 calories in 60 minutes of snowboarding provides a slower caloric burn than some other forms of winter exercise. A 160-pound person burns about 533 calories in an hour of slow-paced cross-country skiing and about 419 calories in 60 minutes of light ice skating. The caloric burn from light snowboarding is equivalent to that of light downhill skiing. Related Reading: How to Estimate the Total Calories You Burned While Running.
results = RAG.search(query="how many calories does skiing virn", k=k) 
print(results)