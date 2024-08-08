from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])




corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")

#### Or provide pretrained sentence-transformer model
model = CrossEncoder("distilroberta-base", num_labels=1, max_length=350)
retriever = TrainRetriever(model=model, batch_size=16)

#### Prepare training samples
train_samples = retriever.load_train(corpus, queries, qrels)
dev_samples = retriever.load_train(dev_corpus, dev_queries, dev_qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

#### Prepare dev evaluator
ir_evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name="nfcorpus-dev")

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v1-{}".format('distilroberta-base', dataset))
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = 1
evaluation_steps = 5000
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

model.fit(
    train_dataloader=train_dataloader,
    evaluator=ir_evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
)