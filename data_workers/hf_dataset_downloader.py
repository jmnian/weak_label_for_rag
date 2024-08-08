from datasets import load_dataset

dataset = load_dataset("microsoft/ms_marco", "v2.1")

dataset['train'].to_csv('/WAVE/datasets/yfang_lab/QA_datasets/msmarco_qa_v2.1_hf/train.csv')
print("train done")
dataset['validation'].to_csv('/WAVE/datasets/yfang_lab/QA_datasets/msmarco_qa_v2.1_hf/validation.csv')
print("val done")
dataset['test'].to_csv('/WAVE/datasets/yfang_lab/QA_datasets/msmarco_qa_v2.1_hf/test.csv')
print("test done")

dont run this, its bad 