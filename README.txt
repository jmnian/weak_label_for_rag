Work flow: 

Step 1. Generate Weak Label
        click weak_label_gen/generate_weak_labels.py -> generate_corpus_queries_qrels will create a folder under "data" fill with corpus, queries, qrels (train, val)
                                 -> generate_weak_labels will create a folder under "data/xxxxxxx" with full_rankxx.jsonl, train_weak.tsv, train_weak_full.tsv 
        run using "python weak_label_gen/generate_weak_labels"

Step 2. Evaluate Weak Label's Quality
        click evaluations/weak_label_quality.py -> specify k values, and the full_rankxxx.jsonl object, to see recall, mrr, and hit rate and all the significance tests 

Step 3. Train Retriever 
    Before start training, make sure to make train_groundtruth_allones.tsv and val.tsv and test.tsv. Scripts are in /data_workers/
        dpr weak -> python train_retriever/train_dpr_weak.py --data_path="/WAVE/users2/unix/jnian/WeakLabelForRAG/data/msmarco_qa_v2_train_corpus500000_weakTrainQ2000_ValQ3000" --weak_label_path="/llama3_old_prompt" --num_hard_neg=10 --num_epochs=16 --encoder_name="bert-base-uncased" --product="cosine" --loss="triplet"
        dpr gt   -> python train_retriever/train_dpr_gt.py --data_path="/WAVE/users2/unix/jnian/WeakLabelForRAG/data/msmarco_qa_v2_train_corpus500000_weakTrainQ2000_ValQ3000" --num_epochs=16 --encoder_name="bert-base-uncased" --product="cosine" --loss="triplet"
        dpr ibneg-> python train_retriever/train_dpr_ibneg.py --product="cosine" --gt_or_weak="weak" --weak_label_path="/llama3_0shot_prompt_top100" --num_epochs=20 --encoder_name="Yibin-Lei/ReContriever" --tsv="train_weak" --data_path="xx" 
        dpr ibneg gt -> python train_retriever/train_dpr_ibneg.py --num_epochs=20 --encoder_name="Yibin-Lei/ReContriever" --product="cosine"  --gt_or_weak="gt" --tsv="train_groundtruth_top1" --data_path="xx"
        ce ibneg -> python train_retriever/train_ce_ibneg.py --product="cosine" --gt_or_weak="weak" --weak_label_path="/llama3_0shot_prompt_top100" --num_epochs=21 --encoder_name="Yibin-Lei/ReContriever" --tsv="train_weak" --data_path="xx"
        ce ibneg gt -> python train_retriever/train_ce_ibneg.py --num_epochs=21 --encoder_name="Yibin-Lei/ReContriever" --product="cosine"  --gt_or_weak="gt" --tsv="train_groundtruth_top1" --data_path="xx"
        ColBERT ibneg 
                -> python train_retriever/train_colbert_ibneg.py --gt_or_weak="weak" --weak_label_path="/llama3_0shot_prompt_top100" --encoder_name="bert-base-uncased" --tsv="train_weak_full" --neg=10 --lr=1e-5 --data_path="/WAVE/users2/unix/jnian/WeakLabelForRAG/data/msmarco_qa_v2_train_corpus500000_weakTrainQ2048_ValQ10000"
        ColBERT ibneg gt 
                -> python train_retriever/train_colbert_ibneg.py --gt_or_weak="gt" --encoder_name="bert-base-uncased" --tsv="train_groundtruth" --neg=10 --lr=1e-5 --data_path="/WAVE/users2/unix/jnian/WeakLabelForRAG/data/msmarco_qa_v2_train_corpus500000_weakTrainQ2048_ValQ10000"
        ColBERT ibneg if not msmarco 
                -> python train_retriever/train_colbert_ibneg.py --gt_or_weak="weak" --weak_label_path="/llama3_0shot_prompt_top100" --encoder_name="bert-base-uncased" --tsv="train_weak" --neg=10 --lr=1e-5 --data_path="xx"
        ColBERT ibneg gt if not msmarco 
                -> python train_retriever/train_colbert_ibneg.py --gt_or_weak="gt" --encoder_name="bert-base-uncased" --tsv="train_groundtruth_top1" --neg=10 --lr=1e-5 --data_path="xx"

Step 4. Evaluate Retriever's Ability on our sampled data: MODEL PATH can be found in train_retriever/output. For ColBERT, look inside .ragatouille
        BM25: 
                go to bm25_evaluation
        DPR: 
                python evaluations/retriever_ability.py --data_path="xx" --model_path="xx"
        ReContriever BiEncoder:
                python evaluations/retriever_ability.py --eval_recontriever --data_path="xx"
        Contriever BiEncoder: 
                python evaluations/retriever_ability.py --eval_contriever --data_path="xx"
        BM25+CE:
                python evaluations/bm25ce_ability.py --bm25_topk=100 --data_path="xx" --model_path="xx" 
        


        Evaluate Retriever's Ability on BEIR 
        DPR:
                python evaluations/retriever_ability_beir.py --beir_dataset_path="/WAVE/datasets/yfang_lab/BEIR" --model_path="/WAVE/users2/unix/jnian/WeakLabelForRAG/train_retriever/output/dpr_weak_2000positive_ibneg_msmarco_qa_v2_train_07_22_22:33" --dataset_name="msmarco"
        Use untrained dpr: 
                python evaluations/retriever_ability_beir.py --beir_dataset_path="/WAVE/datasets/yfang_lab/BEIR" --model_path="bert-base-uncased" --dataset_name="msmarco"
        
Step 5. Evaluate QA Ability (--data could be msmarco/nq/squad/trivia/wq )
        DPR/BM25+CE/Contriever/ReContriever: 
                python evaluations/QA_performance.py --top_k=5 --num_shot=0 --llm_name="llama3" --new_token=20 --use_gt_passage="n" --data="xx" --model_path="xx"
        BM25:
                python evaluations/QA_performance.py --model_path="bm25" --top_k=5 --num_shot=0 --llm_name="llama3" --new_token=20 --use_gt_passage="n" --data="xx"
        Ground truth passage:
                python evaluations/QA_performance.py --llm_name="llama3" --new_token=20 --use_gt_passage="y" --data="xx"
        No Passage: 
                python evaluations/QA_performance.py --top_k=0 --num_shot=0 --llm_name="llama3" --new_token=20 --use_gt_passage="n" --data="xx"

ColBERT Retrieval Ability and QA ability on MSMARCO: (DEFAULT TO USING TOP 1 PASSAGE TO QA)
        (if you already have index, no need to add model_path, other wise, an index will be created for the model and it takes a long time)
        python evaluations/colbert_ability.py --llm_name="llama3" --new_token=20 --data_path="xx" --model_path="xx" --index_name="xx"

To Run experiments for Directly QA using LLM reranked passages that were retrieved from BM25:
        Go to "evaluations/bm25top100_llmRerank_directlyQA.py" and change the "which_experiment" at the top 

To Run Table 5 (Fix prompt study different llm, ms marco on 500 val dataset): 
        Go to "weak_label_gen/generate_weak_labels.py", change parameters in "study_different_llm_on_msmarco" and run "python weak_label_gen/generate_weak_labels.py"
        