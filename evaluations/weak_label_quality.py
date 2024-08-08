import numpy as np
import json 
import os
import csv 
from scipy.stats import ttest_rel

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

def hit_rate_at_k(ranked_list, relevant_set, k):
    for pid, _ in ranked_list[:k]:
        if pid in relevant_set:
            return 1
    return 0


def evaluate_weak_label_quality(rank_dict, levels, data_path):
    '''
    Show llama and bm25 MRR of the best passage, also show the MRR of 2-10 passage (the provided sort of relevant passages)
    do the same for Recall and Hit Rate
    Calculate significance test 
    
    Input: 
        rank_dict, data_path
        
    Output: 
        Just prints some metrics, significance test etc. 
    '''        
    if rank_dict is None: 
        # Load from llama3_x_prompt from data_path
        rank_dict = {}
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                qid = int(data['qid']) # cast "1183942" into 1183942 as the keys 
                rank_dict[qid] = {} 
                rank_dict[qid]["rel_passage"] = data['rel_passage']
                data['bm25_ranked_list'].sort(key=lambda x: x[1], reverse=True)
                data['llm_ranked_list'].sort(key=lambda x: x[1], reverse=True)
                rank_dict[qid]["bm25_ranked_list"] = data['bm25_ranked_list']
                rank_dict[qid]["llm_ranked_list"] = data['llm_ranked_list']
    # if "msmarco" in data_path: 
    #     # Load train_groundtruth.tsv into rel2_10 = {qid: [corpus-id, corpus-id, ...]} all that have scores 0
    #     path_dir = os.path.dirname(data_path) 
    #     base_dir = os.path.dirname(path_dir)  
    #     new_path = os.path.join(base_dir, 'qrels', 'train_groundtruth.tsv')
    #     for qid in rank_dict: 
    #         rank_dict[qid]['rel2_10'] = []
    #     with open(new_path, mode='r', encoding='utf-8') as file:
    #         reader = csv.DictReader(file, delimiter='\t')
    #         for row in reader:
    #             qid = int(row['query-id'])  # Convert query-id to integer
    #             score = int(row['score'])  # Convert score to integer
    #             pid = int(row['corpus-id'])  # Convert corpus-id to integer
    #             if score == 0:
    #                 rank_dict[qid]['rel2_10'].append(pid)
    
    
    results = {
        'recall': {model: {k: [] for k in levels} for model in ['llm', 'bm25']},
        'mrr': {model: [] for model in ['llm', 'bm25']},
        'hit_rate': {model: [] for model in ['llm', 'bm25']}
    }

    # Process each query and calculate metrics
    for qid, info in rank_dict.items():
        relevant_pids = set(info['rel_passage'])
        for model in ['llm_ranked_list', 'bm25_ranked_list']:
            model_key = 'llm' if model == 'llm_ranked_list' else 'bm25'
            ranked_list = sorted(info[model], key=lambda x: x[1], reverse=True)
            
            for k in levels:
                results['recall'][model_key][k].append(recall_at_k(ranked_list, relevant_pids, k))
            results['mrr'][model_key].append(mean_reciprocal_rank(ranked_list, relevant_pids))
            results['hit_rate'][model_key].append(hit_rate_at_k(ranked_list, relevant_pids, 5))  # Assuming HR@5

    # Print out the results and perform statistical tests
    for metric in results:
        if metric == 'recall':
            for k in levels:
                avg_recall_llm = np.mean(results['recall']['llm'][k])
                avg_recall_bm25 = np.mean(results['recall']['bm25'][k])
                print(f"Average Recall@{k} for LLM: {avg_recall_llm:.4f}, BM25: {avg_recall_bm25:.4f}")
                # Statistical test
                t_stat, p_value = ttest_rel(results['recall']['llm'][k], results['recall']['bm25'][k])
                print(f"Recall@{k}: t-statistic = {t_stat}, p-value = {p_value}")
        elif metric in ['mrr', 'hit_rate']:
            avg_metric_llm = np.mean(results[metric]['llm'])
            avg_metric_bm25 = np.mean(results[metric]['bm25'])
            print(f"Average {metric.upper()} for LLM: {avg_metric_llm:.4f}, BM25: {avg_metric_bm25:.4f}")
            # Statistical test
            t_stat, p_value = ttest_rel(results[metric]['llm'], results[metric]['bm25'])
            print(f"{metric.upper()}: t-statistic = {t_stat}, p-value = {p_value}")
            
        
if __name__ == "__main__":
    evaluate_weak_label_quality(None, [1, 5, 10, 20, 50, 100, 120], "/WAVE/users2/unix/jnian/WeakLabelForRAG/data/trivia_train_corpus500000_weakTrainQ2000_ValQ3000/llama3_0shot_prompt_top50/full_rank_lists2000.jsonl")