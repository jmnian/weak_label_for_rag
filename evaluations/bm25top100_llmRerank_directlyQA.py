from generate_weak_labels import load_jsonl, load_or_create_bm25, load_llm, llm_calculate_sequence_log_likelihood, query_likelihood  
import eval_util
import csv, torch
import numpy as np
from tqdm import tqdm 
from nltk.tokenize import word_tokenize

###############################################################################################
data_path = "data/msmarco_qa_v2_train_corpus500000_weakTrainQ2048_ValQ10000" # Dont change
first_stage_ret_count = 100 # most of the time, dont need to change
levels = [1, 5, 10, 20, 50, 100] # feel free to change
rerank_and_qa_llm_name = "llama3" # feel free to change
qa_topk_passage = 1 # Dont change
new_token = 20 # feel free to change
###########################################################################
which_experiment = "wrag_0shot" # "bm25" or "upr" or "wrag_0shot" or "wrag_1shot" or "wrag_2shot" 
###########################################################################


questions, corpus, answers, qrel, weak_and_ground_truth_labels = {}, {}, {}, {}, {}
questions = load_jsonl(questions, f"{data_path}/queries.jsonl")
corpus = load_jsonl(corpus, f"{data_path}/corpus.jsonl")
answers = load_jsonl(answers, f"{data_path}/answers.jsonl")
qrel_path = f"{data_path}/qrels/val.tsv"
with open(qrel_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        qid = int(row['query-id'])
        if row['score'] == '1': 
            if qid not in qrel:
                qrel[qid] = []  # Initialize an empty list for this query-id if it doesn't exist
            qrel[qid].append(int(row['corpus-id']))

bm25 = load_or_create_bm25(corpus, f"{data_path}/bm25_{len(corpus)}.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)
llm, llm_tokenizer = load_llm(rerank_and_qa_llm_name, device, False)

##############
# BM25 retrieve, then llm reranks the passages using different methods 
##############
for qid, rel_pids in tqdm(qrel.items(), desc="LLM Reranking"):
    question, answer = questions[qid], answers[qid]
    ### (1) bm25 retrieve top 100 
    question_tokens = word_tokenize(question.lower())
    passage_scores = bm25.get_scores(question_tokens)
    sorted_indices = np.argsort(passage_scores)[::-1][:first_stage_ret_count]
    top_passages_with_scores = [(list(corpus.keys())[idx], corpus[list(corpus.keys())[idx]], passage_scores[idx]) for idx in sorted_indices]

    weak_and_ground_truth_labels[qid] = {}
    weak_and_ground_truth_labels[qid]["rel_passage"] = rel_pids # why list of rel_pid you may ask, because nq, trivia, wq, squad has multiple relevant passages
    weak_and_ground_truth_labels[qid]["bm25_ranked_list"] = [(pid, score) for pid, _, score in top_passages_with_scores]
    if which_experiment == "bm25":
        continue 
    ### (2) rerank by infer LLM 1 by 1, record score
    llm_score_list = []
    if which_experiment == "upr":
        for pid, passage, _ in top_passages_with_scores:
            log_likelihood = query_likelihood(0, question, passage, answer, llm, llm_tokenizer, device) 
            llm_score_list.append((pid, log_likelihood))
    if which_experiment.startswith("wrag"):
        for pid, passage, _ in top_passages_with_scores:     
            if which_experiment == "wrag_0shot":
                log_likelihood = llm_calculate_sequence_log_likelihood(0, question, passage, answer, llm, llm_tokenizer, device) 
            if which_experiment == "wrag_1shot":
                log_likelihood = llm_calculate_sequence_log_likelihood(1, question, passage, answer, llm, llm_tokenizer, device) 
            if which_experiment == "wrag_2shot":
                log_likelihood = llm_calculate_sequence_log_likelihood(2, question, passage, answer, llm, llm_tokenizer, device) 
            llm_score_list.append((pid, log_likelihood))
    llm_ranked_list = sorted(llm_score_list, key=lambda x: x[1], reverse=True)
    weak_and_ground_truth_labels[qid]["llm_ranked_list"] = llm_ranked_list 


##############
# Evaluate Retrieval Performance
##############
def recall_at_k(ranked_list, relevant_set, k):
    relevant_retrieved = 0
    for pid, _ in ranked_list[:k]:
        if pid in relevant_set:
            relevant_retrieved += 1
    return relevant_retrieved / len(relevant_set)

def mean_reciprocal_rank(ranked_list, relevant_set):
    for index, (pid, _) in enumerate(ranked_list, start=1):
        if pid in relevant_set:
            return 1 / index
    return 0

results = {
    'recall': {model: {k: [] for k in levels} for model in ['llm', 'bm25']},
    'mrr': {model: [] for model in ['llm', 'bm25']},
}

# Process each query and calculate metrics
for qid, info in weak_and_ground_truth_labels.items():
    relevant_pids = set(info['rel_passage'])
    lists = ['llm_ranked_list', 'bm25_ranked_list'] if which_experiment != 'bm25' else ['bm25_ranked_list']
    for model in lists:
        model_key = 'llm' if model == 'llm_ranked_list' else 'bm25'
        ranked_list = sorted(info[model], key=lambda x: x[1], reverse=True)
        
        for k in levels:
            results['recall'][model_key][k].append(recall_at_k(ranked_list, relevant_pids, k))
        results['mrr'][model_key].append(mean_reciprocal_rank(ranked_list, relevant_pids))

# Print out the results and perform statistical tests
for metric in results:
    if metric == 'recall':
        for k in levels:
            avg_recall_llm = np.mean(results['recall']['llm'][k])
            avg_recall_bm25 = np.mean(results['recall']['bm25'][k])
            print(f"Average Recall@{k} for LLM: {avg_recall_llm:.4f}, BM25: {avg_recall_bm25:.4f}")
    else:
        avg_metric_llm = np.mean(results[metric]['llm'])
        avg_metric_bm25 = np.mean(results[metric]['bm25'])
        print(f"Average {metric.upper()} for LLM: {avg_metric_llm:.4f}, BM25: {avg_metric_bm25:.4f}")
        
##############
# Evaluate QA Performance ONLY WORKS FOR MSMARCO because I assume there is only 1 answer to each question
##############

def one_passage_0shot_prompt(passage, question):
    prompt = f'''PASSAGE: {passage} 
QUESTION: {question}
Answer the user's QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT does not contain the facts to answer the QUESTION, forget about the DOCUMENTS and directly answer the QUESTION
Keep the answer within one short sentence. 
ANSWER: '''
    return prompt 

def one_passage_1shot_prompt(passage, question): 
    prompt = f'''Example 1:
PASSAGE: If you have arthritis, you may have considered a cortisone shot as part of your treatment plan. These shots are not pain relievers. Cortisone is a type of steroid, a drug that lowers inflammation, which is something that can lead to less pain. Cortisone injections can be used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon. They can also treat inflammation that is widespread throughout the body, such as with allergic reactions, asthma, and rheumatoid arthritis, which affects many joints.
QUESTION: what is a cortisone injection used for
ANSWER: Used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon.

PASSAGE: {passage} 
QUESTION: {question}
Answer the user's QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT does not contain the facts to answer the QUESTION, forget about the DOCUMENTS and directly answer the QUESTION
Keep the answer within one short sentence. 
ANSWER: '''
    return prompt 

def one_passage_2shot_prompt(passage, question):
    prompt = f'''Example 1:
PASSAGE: If you have arthritis, you may have considered a cortisone shot as part of your treatment plan. These shots are not pain relievers. Cortisone is a type of steroid, a drug that lowers inflammation, which is something that can lead to less pain. Cortisone injections can be used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon. They can also treat inflammation that is widespread throughout the body, such as with allergic reactions, asthma, and rheumatoid arthritis, which affects many joints.
QUESTION: what is a cortisone injection used for
ANSWER: Used to treat inflammation of small areas of the body, such as inflammation of a specific joint or tendon.

Example 2: 
PASSAGE: September Weather in Paris. – min temperature : 11°C / 52°F. – max temperature : 21°C / 70°F. – average rainfall : 54 mm. Although the Parisian weather can be quite variable in September, it remains pleasant for the most part. September is generally a sunny month with one of the lowest average rainfalls in the year.
QUESTION: average temperature in paris in september in fahrenheit
ANSWER: Min temperature: 52°F and max temperature: 70°F.

PASSAGE: {passage} 
QUESTION: {question}
Answer the user's QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT does not contain the facts to answer the QUESTION, forget about the DOCUMENTS and directly answer the QUESTION
Keep the answer within one short sentence. 
ANSWER: '''
    return prompt 

answers_tobe_eval = []
for qid, item in weak_and_ground_truth_labels.items():
    if which_experiment == 'bm25':
        ranked_list = sorted(item['bm25_ranked_list'], key=lambda x: x[1], reverse=True)
    else: 
        ranked_list = sorted(item['llm_ranked_list'], key=lambda x: x[1], reverse=True)
    question, answer, passage = questions[qid], answers[qid], corpus[ranked_list[0][0]]

    prompt = one_passage_0shot_prompt(passage, question)
        
    inputs = llm_tokenizer([prompt], return_tensors='pt').to(llm.device)
    output = llm.generate(**inputs, max_new_tokens=new_token)
    actual_output = output[0][-new_token:]
    answer_gen = llm_tokenizer.decode(actual_output, skip_special_tokens=True)
    a, g = eval_util.normalize_answer(answer), eval_util.normalize_answer(answer_gen)
    answers_tobe_eval.append(([a], g, qid))
    
print(f"Results for {which_experiment}, {qa_topk_passage} passage per prompt")
eval_util.evaluate(answers_tobe_eval)
