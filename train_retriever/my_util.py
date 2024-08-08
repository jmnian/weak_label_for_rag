import json
import os
import csv


def make_hard_neg_tsv(input_path, x):

    # Find the full_rank .jsonl file
    for filename in os.listdir(input_path):
        if filename.startswith("full_rank") and filename.endswith(".jsonl"):
            full_path = os.path.join(input_path, filename)
            break

    outfile_path = os.path.join(input_path, f"{input_path}/train_weak_{x}.tsv")
    
    if os.path.exists(outfile_path):
        print(f"Already have {outfile_path}")
        return outfile_path
    
    # Read the .jsonl file and process each line
    with open(full_path, 'r') as infile, open(outfile_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        writer.writerow(['query-id', 'corpus-id', 'score'])

        for line in infile:
            data = json.loads(line)
            query_id = data['qid']
            llm_ranked_list = data.get('llm_ranked_list', [])

            if llm_ranked_list:
                # Write the first tuple with score of 1
                writer.writerow([query_id, llm_ranked_list[0][0], 1])

                # Write the next x tuples with score of 0
                for pid, _ in llm_ranked_list[1:x+1]:  # Ensure not to exceed the list length
                    writer.writerow([query_id, pid, 0])

    print(f"Hard neg tsv ({x} per positive) successfully created: {outfile_path}")
    return outfile_path