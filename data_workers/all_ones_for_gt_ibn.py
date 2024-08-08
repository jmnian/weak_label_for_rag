import pandas as pd

# Path to the input TSV file
input_path = '/WAVE/users2/unix/jnian/WeakLabelForRAG/data/nq_train_corpus500000_weakTrainQ2000_ValQ3000/qrels/train_groundtruth.tsv'

# Load the data from the TSV file
data = pd.read_csv(input_path, delimiter='\t')

# Filter rows where the 'score' column is 1
filtered_data = data[data['score'] == 1]

# Path to save the output TSV file
output_path = '/WAVE/users2/unix/jnian/WeakLabelForRAG/data/nq_train_corpus500000_weakTrainQ2000_ValQ3000/qrels/train_groundtruth_allones.tsv'

# Save the filtered data to a new TSV file
filtered_data.to_csv(output_path, sep='\t', index=False)

print(f"Filtered data saved to {output_path}")
