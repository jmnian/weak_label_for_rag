import pandas as pd

def filter_qrels_train_groundtruth(input_tsv, output_tsv, k):
    qrel_df = pd.read_csv(input_tsv, sep='\t', header=None, names=["query-id", "corpus-id", "score"])
    filtered_qrel_df = qrel_df.groupby('query-id').head(k)
    filtered_qrel_df.to_csv(output_tsv, sep='\t', header=False, index=False)

def filter_qrels_train_weak(input_tsv, output_tsv, k):
    qrel_df = pd.read_csv(input_tsv, sep='\t', header=None, names=["query-id", "corpus-id", "score"])
    filtered_qrel_df = qrel_df.groupby('query-id').head(k).reset_index(drop=True)
    filtered_qrel_df.loc[:, 'score'] = 1  # Set all scores to 1 using .loc to avoid SettingWithCopyWarning
    filtered_qrel_df.to_csv(output_tsv, sep='\t', header=False, index=False)

# Example usage
k = 5
input_tsv = "/WAVE/users2/unix/jnian/WeakLabelForRAG/data/squad_train_corpus500000_weakTrainQ2000_ValQ3000/llama3_0shot_prompt_top100/train_weak_full.tsv"
output_tsv = f"/WAVE/users2/unix/jnian/WeakLabelForRAG/data/squad_train_corpus500000_weakTrainQ2000_ValQ3000/llama3_0shot_prompt_top100/train_weak_top{k}.tsv"
# filter_qrels_train_groundtruth(input_tsv, output_tsv, k)
filter_qrels_train_weak(input_tsv, output_tsv, k)