from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from tqdm import tqdm
import pathlib, os, argparse, datetime 
import logging
import random
random.seed(42)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

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

model = CrossEncoder(args.encoder_name, num_labels=1, max_length=350)

#### Prepare training samples
# retriever = TrainRetriever(model=model, batch_size=32)
# train_samples = retriever.load_train(corpus, queries, qrels)
# dev_samples = retriever.load_train(dev_corpus, dev_queries, dev_qrels)
# train_dataloader = retriever.prepare_train(train_samples, shuffle=True)


# Prepare training data. Here we want to take each relevant qd pair, add it to train_samples as positive, 
# sample some other qd pairs, using those d as negative. We do this for all relevant qd pair, resulting in 2000 * (neg_per_pos + 1) training samples
##################
neg_per_pos = 9 
dev_neg_per_pos = 100
##################
corpus_keys = list(corpus.keys())
train_samples = []
for qid, pos_dict in qrels.items():    # 913821, {'3': 1}
    query = queries[qid]
    neg_count = 0
    for pos_pid in pos_dict.keys():
        train_samples.append(InputExample(texts=[query, corpus[pos_pid]['text']], label=1))
    while neg_count < neg_per_pos:
        neg_pid = random.choice(corpus_keys)
        while neg_pid == pos_pid:
            neg_pid = random.choice(corpus_keys)
        train_samples.append(InputExample(texts=[query, corpus[neg_pid]['text']], label=0))
        neg_count += 1
# assert len(train_samples) == len(qrels) * (neg_per_pos + 1)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)

dev_corpus_keys = list(dev_corpus.keys())
dev_samples = {}
for dev_qid, dev_pos_dict in dev_qrels.items():
    dev_query = dev_queries[dev_qid]
    dev_samples[dev_qid] = {"query": dev_query, "positive": set(), "negative": set()}
    for dev_pos_pid in dev_pos_dict.keys():
        dev_samples[dev_qid]["positive"].add(dev_corpus[dev_pos_pid]["text"])
    dev_neg_count = 0
    while dev_neg_count < dev_neg_per_pos * len(dev_pos_dict):
        dev_neg_pid = random.choice(corpus_keys)
        while dev_neg_pid == dev_pos_pid:
            dev_neg_pid = random.choice(corpus_keys)
        dev_samples[dev_qid]["negative"].add(corpus[dev_neg_pid]["text"])
        dev_neg_count += 1
        



#### Prepare dev evaluator
ir_evaluator = CERerankingEvaluator(dev_samples)

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
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"ce_weak_ibneg_{dataset}_{encoder}_{args.tsv}_{time_string}")
elif args.gt_or_weak == "gt":
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"ce_gt_ibneg_{dataset}_{encoder}_{args.tsv}_{time_string}")
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = args.num_epochs
evaluation_steps = -1
warmup_steps = 0
# warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

model.fit(
    train_dataloader=train_dataloader,
    evaluator=ir_evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    use_amp=True,
)