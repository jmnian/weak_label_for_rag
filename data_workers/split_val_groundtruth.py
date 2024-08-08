import pandas as pd
import numpy as np
import os 


input_path = '/WAVE/users2/unix/jnian/WeakLabelForRAG/data/wq_train_corpus163683_weakTrainQ2000_ValQ474/qrels/val_groundtruth.tsv'
val_size = 1000 # test size is not initialized, validation size would just be whatever is left after splitting out the test samples . 



directory_path = os.path.dirname(input_path)
# Load the data
data = pd.read_csv(input_path, delimiter='\t')

# Group by query-id to handle multiple entries per query-id
grouped = data.groupby('query-id')

# Generate a list of unique query-ids
unique_query_ids = list(grouped.groups.keys())

# Shuffle the list to randomize selection
np.random.shuffle(unique_query_ids)

# Split the query-ids into test and validation sets
test_query_ids = unique_query_ids[val_size:] 
val_query_ids = unique_query_ids[:val_size]    # first val_size queries
 


# Select rows for test and validation based on query-ids
test_data = data[data['query-id'].isin(test_query_ids)]
val_data = data[data['query-id'].isin(val_query_ids)]

# Path to save the output TSV files
test_output_path = directory_path + '/test.tsv'
val_output_path = directory_path + '/val.tsv'

# Save the test and validation data to separate TSV files
test_data.to_csv(test_output_path, sep='\t', index=False)
val_data.to_csv(val_output_path, sep='\t', index=False)

print(f"Test data saved to {test_output_path}")
print(f"Validation data saved to {val_output_path}")