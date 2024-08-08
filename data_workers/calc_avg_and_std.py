import json
import numpy as np

# Function to count the number of words in a string
def count_words(text):
    return len(text.split())

# Read JSON data from file
file_path = '/WAVE/users2/unix/jnian/WeakLabelForRAG/data/wq_train_corpus163683_weakTrainQ2000_ValQ474/answers.jsonl'
with open(file_path, 'r') as file:
    json_lines = file.readlines()

# Parse JSON data and extract word counts
word_counts = [count_words(json.loads(line)['text']) for line in json_lines]

# Calculate average number of words
average_word_count = np.mean(word_counts)

# Calculate standard deviation of word counts
std_deviation = np.std(word_counts)

# Print the results
print(f"Average Number of Words: {average_word_count}")
print(f"Standard Deviation: {std_deviation}")
