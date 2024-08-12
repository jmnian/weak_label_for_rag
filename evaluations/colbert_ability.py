from ragatouille import RAGPretrainedModel
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm 
import argparse, logging, json, os, torch, eval_util

parser = argparse.ArgumentParser(description="Script for training a retriever")
parser.add_argument('--data_path', type=str, help='Path to the corpus, queries, etc.')
parser.add_argument('--model_path', type=str, help='Path to the model that you want to evaluate')
parser.add_argument('--llm_name', type=str, help="Which LLM to use to answer the question")
parser.add_argument('--new_token', type=int, help="Number of new token to allow llm to generate")
parser.add_argument('--index_name', type=str, help="Give the index a name, or try to load an existing index")
args = parser.parse_args()

# Either load index or create index from a trained ColBERT
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
        use_faiss=True # doesn't use gpu for some reason. I heard if you create conda env and install faiss-gpu it works and it's a lot faster. 
        )


##########################################################
# k = 3 
# # ground truth passage should be: Cross-Country Skiing Burns More. Burning about 381 calories in 60 minutes of snowboarding provides a slower caloric burn than some other forms of winter exercise. A 160-pound person burns about 533 calories in an hour of slow-paced cross-country skiing and about 419 calories in 60 minutes of light ice skating. The caloric burn from light snowboarding is equivalent to that of light downhill skiing. Related Reading: How to Estimate the Total Calories You Burned While Running.
# colbert_results = RAG.search(query="how many calories does skiing virn", k=k) 
# print(colbert_results)
'''
colbert_results = [{'content': "the passage", 'score': 21.25, 'rank': 1, 'document_id': 'some hash', 'passage_id': 20450}]
'''
########################################################

'''
Prepare "results" into BEIR format and we are good to go 
results = {qid: {doc_id: score}, ...}
'''
corpus, queries, qrels = GenericDataLoader(args.data_path).load(split="test")

def find_doc_id_by_text(corpus, search_text):
    for doc_id, content in corpus.items():
        if content.get('text') == search_text or search_text in content.get('text'):
            return str(doc_id)
    print(f"Cant't find:{search_text}")
    return None


results_file = f"evaluations/colbert_retrieval_results/{args.index_name}_retrieval_resuls.json"
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    pqa_list = []
    for qid, rank_dict in results.items():
        query = queries[qid]
        answer = answers[qid]
        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        if sorted_docs:
            top_doc_id = sorted_docs[0][0]                           # THIS IS WHERE I ONLY TAKE TOP 1 TO DO QA
            top_doc_text = corpus[top_doc_id]['text']
            pqa_list.append(([top_doc_text], query, answer, qid))
else:
    results, pqa_list = {}, []
    answers = eval_util.load_jsonl({}, f"{args.data_path}/answers.jsonl")

    for qid, rel_doc in tqdm(qrels.items(), desc="ColBERT retrieving"): 
        results[qid] = {}
        query, answer = queries[qid], answers[qid]
        colbert_results = RAG.search(query=query, k=100)
        topk = 1
        for doc_dict in colbert_results:
            doc, score = doc_dict['content'], doc_dict['score']
            doc_id = find_doc_id_by_text(corpus, doc)  # I know this is super slow. 
            results[qid][doc_id] = score
            if topk > 0:
                pqa_list.append(([doc], query, answer, qid))        # THIS IS WHERE I ONLY TAKE TOP 1 TO DO QA
                topk -= 1
    # Save results to a JSON file
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

# Evaluate Retrieval ability
evaluator = EvaluateRetrieval(k_values=[1,3,5,10,100])
ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, evaluator.k_values)
mrr = evaluator.evaluate_custom(qrels, results, evaluator.k_values, metric="mrr")

print(f"Retrieval results of {args.index_name}")
print(recall)
print(mrr)


# DOING QA and then evaluate 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm, tokenizer = eval_util.load_llm(args.llm_name, device, False)
answers_tobe_eval = []

for passages, question, answer, qid in tqdm(pqa_list, desc="Generating Answers"):
    prompt = eval_util.one_passage_0shot_prompt(passages, question)
    inputs = tokenizer([prompt], return_tensors='pt').to(llm.device)
    output = llm.generate(**inputs, max_new_tokens=args.new_token)
    actual_output = output[0][-args.new_token:]
    answer_gen = tokenizer.decode(actual_output, skip_special_tokens=True)
    a, g = eval_util.normalize_answer(answer), eval_util.normalize_answer(answer_gen)
    answers_tobe_eval.append(([a], g, qid))

output_file_name = f"evaluations/qa/msmarco/colbert_{args.index_name}"
eval_util.evaluate(answers_tobe_eval, output_file_name, args.new_token)