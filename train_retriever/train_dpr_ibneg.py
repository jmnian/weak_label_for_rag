'''
This examples show how to train a basic Bi-Encoder for any BEIR dataset without any mined hard negatives or triplets.

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass pairs in the format:
(query, positive_passage). Other positive passages within a single batch becomes negatives given the pos passage.

We do not mine hard negatives or train triplets in this example.

Running this script:
python train_sbert.py
'''

from sentence_transformers import losses, models, SentenceTransformer
from sentence_transformers.readers import InputExample
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from collections import defaultdict
import pathlib, os, argparse, datetime 
import logging
import random
random.seed(42)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser(description="Script for training a retriever")
parser.add_argument('--data_path', type=str, help='Path to the corpus, queries, etc.')
parser.add_argument('--weak_label_path', type=str, help='Path to the weak label data file')
parser.add_argument('--num_epochs', type=int, help='Training epochs')
parser.add_argument('--encoder_name', type=str, help='What model to use as the encoder of the retriever')
parser.add_argument('--product', type=str, help='Train using cosine product or dot product')
parser.add_argument('--gt_or_weak', type=str, help='Use Ground truth data to train or not')
parser.add_argument('--tsv', type=str, help='When using Ground truth or Weak data to train, which tsv to use')
args = parser.parse_args()

if args.gt_or_weak == "gt":
    print("Using ground truth data to train, in-batch negative style")
    corpus, queries, qrels = GenericDataLoader(data_folder=args.data_path).load(split=args.tsv)
elif args.gt_or_weak == "weak":
    print("Using llm weak label to train, in-batch negative style")
    corpus, queries, qrels = GenericDataLoader(data_folder=args.data_path, 
                                            qrels_folder=args.data_path+args.weak_label_path).load(split=args.tsv)
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(args.data_path).load(split="val")



#### Provide any sentence-transformers or HF model
model_name = args.encoder_name
word_embedding_model = models.Transformer(model_name, max_seq_length=350)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#### Or provide pretrained sentence-transformer model
# model = SentenceTransformer("msmarco-distilbert-base-v3")

retriever = TrainRetriever(model=model, batch_size=128)

def convert_train_samples(train_samples):
    '''
    train_samples: [InputExample] 
    return: [(query, [doc1, doc2, doc3, ...])]
    '''
    query_to_docs = defaultdict(list)
    for item in train_samples:
        query = item.texts[0]
        document = item.texts[1]
        query_to_docs[query].append(document)
    grouped_samples = [(query, docs) for query, docs in query_to_docs.items()]
    
    return grouped_samples
    
    
#### Prepare training samples
# train_samples = retriever.load_train(corpus, queries, qrels)
# train_samples = convert_train_samples(train_samples)
# train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

train_samples = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

if args.product == "cosine":
    #### Training SBERT with cosine-product
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
elif args.product == "dot":
    #### training SBERT with dot-product
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

#### Prepare dev evaluator
ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

#### If no dev set is present from above use dummy evaluator
# ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
if args.encoder_name == "bert-base-uncased":
    encoder = "BERT"
elif args.encoder_name == "Yibin-Lei/ReContriever":
    encoder = "ReContriever"
elif args.encoder_name == "facebook/contriever":
    encoder = "Contriever"

dataset = args.data_path.split("_corpus")[0].split("/")[-1]
current_time = datetime.datetime.now()
time_string = current_time.strftime('%m_%d_%H:%M')
if args.gt_or_weak == "weak":
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"dpr_weak_ibneg_{dataset}_{encoder}_{args.tsv}_{time_string}")
elif args.gt_or_weak == "gt":
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"dpr_gt_ibneg_{dataset}_{encoder}_{args.tsv}_{time_string}")
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = args.num_epochs
evaluation_steps = -1
# warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)
warmup_steps = 0 




retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)