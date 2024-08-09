import json
from scipy.stats import ttest_rel, wilcoxon 
import numpy as np

baseline_file = "/WAVE/users2/unix/jnian/WeakLabelForRAG/evaluations/qa/msmarco/ReContriever__08_08_11:43_20token.json"
improved_file = "evaluations/qa/msmarco/dpr_weak_ibneg_msmarco_qa_v2_train_ReContriever_top1_08_05_23:07__08_09_11:32_20token.json"

with open(baseline_file, 'r') as f:
    baseline_data = json.load(f)

with open(improved_file, 'r') as f:
    improved_data = json.load(f)

# Extract metrics
metrics = ["f1", "rouge-1", "rouge-2", "rouge-l", "bleu", "bleu-1", "meteor", "em"]
results = {}

for metric in metrics:
    baseline_values = [baseline_data[key][metric] for key in baseline_data]
    improved_values = [improved_data[key][metric] for key in improved_data]
    
    # Perform paired t-test
    t_stat, p_value_ttest = ttest_rel(baseline_values, improved_values)
    
    # Perform Wilcoxon signed-rank test
    w_stat, p_value_wilcoxon = wilcoxon(baseline_values, improved_values)
    
    results[metric] = {
        "t_stat": t_stat,
        "p_value_ttest": p_value_ttest,
        "w_stat": w_stat,
        "p_value_wilcoxon": p_value_wilcoxon,
        "baseline_mean": np.mean(baseline_values),
        "improved_mean": np.mean(improved_values)
    }

# Print results
for metric in results:
    print(f"Metric: {metric}")
    print(f"  Baseline Mean: {results[metric]['baseline_mean']}")
    print(f"  Improved Mean: {results[metric]['improved_mean']}")
    print(f"  Paired t-test p-value: {results[metric]['p_value_ttest']}")
    print(f"  Wilcoxon test p-value: {results[metric]['p_value_wilcoxon']}")