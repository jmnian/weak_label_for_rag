'''
This examples show how to train a Bi-Encoder using BEIR format data.

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, sampled from llm reranking. Hard negatives can be at most 100 per positive sample 
'''

from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.train import TrainRetriever
import pathlib, os, tqdm, csv, argparse, datetime 
import logging
from my_util import make_hard_neg_tsv


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser(description="Script for training a retriever")
parser.add_argument('--data_path', type=str, help='Path to the corpus, queries, etc.')
parser.add_argument('--num_epochs', type=int, help='Training epochs')
parser.add_argument('--encoder_name', type=str, help='What model to use as the encoder of the retriever')
parser.add_argument('--product', type=str, help='Train using cosine product or dot product')
parser.add_argument('--loss', type=str, help='What Loss function to use')
args = parser.parse_args()

data_path = args.data_path # /WAVE/users2/unix/jnian/WeakLabelForRAG/data/msmarco_qa_v2_train_corpus500000_weakTrainQ1200_ValQ10000

#### Provide the data_path for corpus, queries, and qrel 
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train_groundtruth")

#### Please Note not all datasets contain a dev split, comment out the line if such the case
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="val_groundtruth")

#### Construct Triplets
triplets = []
for qid, rels in tqdm.tqdm(qrels.items(), desc="Construct Triplets"):
    query_text = queries[qid]
    pos_text = ""
    for cid, score in rels.items(): # select the pos text
        if score == 1: 
            pos_text = corpus[cid]['text']
    for cid, score in rels.items():
        if score == 0: 
            neg_text = corpus[cid]['text']
            triplets.append([query_text, pos_text, neg_text])

#### Provide any sentence-transformers or HF model
model_name = args.encoder_name
word_embedding_model = models.Transformer(model_name, max_seq_length=300)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#### Provide a high batch-size to train better with triplets!
retriever = TrainRetriever(model=model, batch_size=32)

#### Prepare triplets samples
train_samples = retriever.load_train_triplets(triplets=triplets)
train_dataloader = retriever.prepare_train_triplets(train_samples)

# if args.product == "cosine":
#     #### Training SBERT with cosine-product
#     train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
# elif args.product == "dot":
#     #### training SBERT with dot-product
#     train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

if args.loss == "triplet": 
    train_loss = losses.TripletLoss(model=retriever.model)

#### Prepare dev evaluator
ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

#### If no dev set is present from above use dummy evaluator
# ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
dataset = data_path.split("_corpus")[0].split("/")[-1]
current_time = datetime.datetime.now()
time_string = current_time.strftime('%m_%d_%H:%M')
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"dpr_gt_{len(qrels)}positive_{dataset}_{time_string}")
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = args.num_epochs
evaluation_steps = -1
warmup_steps = 0
optimizer_params = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}
# warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=model_save_path,
                optimizer_params=optimizer_params,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)





