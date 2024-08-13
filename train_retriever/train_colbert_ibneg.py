from ragatouille import RAGTrainer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
import pathlib, os, argparse, datetime, logging, random 
random.seed(42)

if __name__ == "__main__": # If i don't add this, the multiprocessing module complains. 
    
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    parser = argparse.ArgumentParser(description="Script for training ColBERT")
    parser.add_argument('--data_path', type=str, help='Path to the corpus, queries, etc.')
    parser.add_argument('--weak_label_path', type=str, help='Path to the weak label data file')
    parser.add_argument('--num_epochs', type=int, help='Training epochs')
    parser.add_argument('--encoder_name', type=str, help='What model to use as the encoder of the retriever')
    parser.add_argument('--gt_or_weak', type=str, help='Use Ground truth data to train or not')
    parser.add_argument('--tsv', type=str, help='When using Ground truth or Weak data to train, which tsv to use')
    parser.add_argument('--neg', type=int, help='How many negatives per positive')
    parser.add_argument('--lr', type=float, help='ColBERT Learning rate')
    args = parser.parse_args()



    if args.gt_or_weak == "gt":
        print("Using ground truth data to train, in-batch negative style")
        corpus, queries, qrels = GenericDataLoader(data_folder=args.data_path).load(split=args.tsv)
    elif args.gt_or_weak == "weak":
        print("Using llm weak label to train, in-batch negative style")
        corpus, queries, qrels = GenericDataLoader(data_folder=args.data_path, 
                                                qrels_folder=args.data_path+args.weak_label_path).load(split=args.tsv)
        
    # Make pos_pairs = [[query, passage, 1], [query, passage, 0], ...]
    pairs = []
    hard_neg_per_pos = args.neg 
    for qid, rel in qrels.items():
        has_pos = False
        num_neg = hard_neg_per_pos
        query = queries[qid]
        for pid, score in rel.items():
            if num_neg == 0 and has_pos: 
                break 
            if score == 1:
                positive = corpus[pid]['text'] + ' ' + corpus[pid]['title']
                pairs.append([query, positive, 1])
                has_pos = True
            elif score == 0 and num_neg > 0: 
                negative = corpus[pid]['text'] + ' ' + corpus[pid]['title']
                pairs.append([query, negative, 0])
                num_neg -= 1 


    trainer = RAGTrainer(model_name=f"colbert_{args.gt_or_weak}_{args.data_path.split('/')[-1]}", 
                            pretrained_model_name = args.encoder_name,
                            language_code='en',
                            n_usable_gpus=1
                        )
    trainer.prepare_training_data(raw_data=pairs,
                                data_out_path='./colbert_data/',
                                num_new_negatives=0, # ragatouille mines negatives for you, or randomly sample
                                hard_negative_minimum_rank=0, # this shouldn't do anything
                                mine_hard_negatives=False,
                                hard_negative_model_size='small', # this shouldn't do anything
                                pairs_with_labels=True,
                                positive_label=1,
                                negative_label=0
                                )
    model_path = trainer.train( batch_size=64, 
                                nbits=4, 
                                maxsteps=500000,
                                use_ib_negatives=True,  # using in batch negatives 
                                learning_rate=args.lr,
                                dim=128,
                                doc_maxlen=350,
                                use_relu=True,
                                warmup_steps=0, # 'auto' for 10% of total steps
                                accumsteps=1
                                )