import logging
import numpy as np
import os
import csv
from tqdm import tqdm 

logger = logging.getLogger(__name__)

# class CERerankingEvaluator:
#     """
#     This class evaluates a CrossEncoder model for the task of re-ranking.

#     Given a query and a list of documents, it computes the score [query, doc_i] for all possible
#     documents and sorts them in decreasing order. Then, MRR@5, Recall@1, and Recall@5 are computed to measure the quality of the ranking.

#     :param samples: Must be a list and each element is of the form: {'query': '', 'positive': [], 'negative': []}. Query is the search query,
#      positive is a list of positive (relevant) documents, negative is a list of negative (irrelevant) documents.
#     """
#     def __init__(self, samples, name: str = '', write_csv: bool = True):
#         self.samples = samples
#         self.name = name
#         self.mrr_at_k = 5

#         if isinstance(self.samples, dict):
#             self.samples = list(self.samples.values())

#         self.csv_file = "CERerankingEvaluator" + ("_" + name if name else '') + "_results.csv"
#         self.csv_headers = ["epoch", "steps", "MRR@5", "Recall@1", "Recall@5"]
#         self.write_csv = write_csv

#     def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
#         if epoch != -1:
#             if steps == -1:
#                 out_txt = " after epoch {}:".format(epoch)
#             else:
#                 out_txt = " in epoch {} after {} steps:".format(epoch, steps)
#         else:
#             out_txt = ":"

#         logger.info("CERerankingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

#         all_mrr_scores = []
#         all_recall_at_1 = []
#         all_recall_at_5 = []
#         num_queries = 0
#         num_positives = []
#         num_negatives = []

#         for instance in tqdm(self.samples, desc="Evaluating", unit="sample"):
#             query = instance['query']
#             positive = list(instance['positive'])
#             negative = list(instance['negative'])
#             docs = positive + negative
#             is_relevant = [True]*len(positive) + [False]*len(negative)

#             if len(positive) == 0 or len(negative) == 0:
#                 continue

#             num_queries += 1
#             num_positives.append(len(positive))
#             num_negatives.append(len(negative))

#             model_input = [[query, doc] for doc in docs]
#             pred_scores = model.predict(model_input, convert_to_numpy=True, show_progress_bar=False)
#             pred_scores_argsort = np.argsort(-pred_scores)  # Sort in decreasing order

#             # Calculate MRR@5
#             mrr_score = 0
#             for rank, index in enumerate(pred_scores_argsort[0:self.mrr_at_k]):
#                 if is_relevant[index]:
#                     mrr_score = 1 / (rank + 1)
#                     break
#             all_mrr_scores.append(mrr_score)

#             # Calculate Recall@1
#             recall_at_1 = 1 if is_relevant[pred_scores_argsort[0]] else 0
#             all_recall_at_1.append(recall_at_1)

#             # Calculate Recall@5
#             recall_at_5 = sum(is_relevant[i] for i in pred_scores_argsort[0:5]) / len(positive)
#             all_recall_at_5.append(recall_at_5)

#         mean_mrr = np.mean(all_mrr_scores)
#         mean_recall_at_1 = np.mean(all_recall_at_1)
#         mean_recall_at_5 = np.mean(all_recall_at_5)

#         logger.info("Queries: {} \t Positives: Min {:.1f}, Mean {:.1f}, Max {:.1f} \t Negatives: Min {:.1f}, Mean {:.1f}, Max {:.1f}".format(
#             num_queries, np.min(num_positives), np.mean(num_positives), np.max(num_positives),
#             np.min(num_negatives), np.mean(num_negatives), np.max(num_negatives)
#         ))
#         logger.info("MRR@{}: {:.2f}".format(self.mrr_at_k, mean_mrr * 100))
#         logger.info("Recall@1: {:.2f}".format(mean_recall_at_1 * 100))
#         logger.info("Recall@5: {:.2f}".format(mean_recall_at_5 * 100))

#         if output_path is not None and self.write_csv:
#             csv_path = os.path.join(output_path, self.csv_file)
#             output_file_exists = os.path.isfile(csv_path)
#             with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 if not output_file_exists:
#                     writer.writerow(self.csv_headers)

#                 writer.writerow([epoch, steps, mean_mrr, mean_recall_at_1, mean_recall_at_5])

#         return mean_recall_at_5 # score to look at when saving model weights

class CERerankingEvaluator:
    """
    This class evaluates a CrossEncoder model for the task of re-ranking.

    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 is compute to measure the quality of the ranking.

    :param samples: Must be a list and each element is of the form: {'query': '', 'positive': [], 'negative': []}. Query is the search query,
     positive is a list of positive (relevant) documents, negative is a list of negative (irrelevant) documents.
    """
    def __init__(self, samples, mrr_at_k: int = 5, name: str = '', write_csv: bool = True):
        self.samples = samples
        self.name = name
        self.mrr_at_k = mrr_at_k

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        self.csv_file = "CERerankingEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "MRR@{}".format(mrr_at_k)]
        self.write_csv = write_csv

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CERerankingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        all_mrr_scores = []
        num_queries = 0
        num_positives = []
        num_negatives = []
        for instance in tqdm(self.samples, desc="Evaluating", unit="sample"):
            query = instance['query']
            positive = list(instance['positive'])
            negative = list(instance['negative'])
            docs = positive + negative
            is_relevant = [True]*len(positive) + [False]*len(negative)

            if len(positive) == 0 or len(negative) == 0:
                continue

            num_queries += 1
            num_positives.append(len(positive))
            num_negatives.append(len(negative))

            model_input = [[query, doc] for doc in docs]
            pred_scores = model.predict(model_input, convert_to_numpy=True, show_progress_bar=False)
            pred_scores_argsort = np.argsort(-pred_scores)  #Sort in decreasing order

            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0:self.mrr_at_k]):
                if is_relevant[index]:
                    mrr_score = 1 / (rank+1)
                    break

            all_mrr_scores.append(mrr_score)

        mean_mrr = np.mean(all_mrr_scores)
        logger.info("Queries: {} \t Positives: Min {:.1f}, Mean {:.1f}, Max {:.1f} \t Negatives: Min {:.1f}, Mean {:.1f}, Max {:.1f}".format(num_queries, np.min(num_positives), np.mean(num_positives), np.max(num_positives), np.min(num_negatives), np.mean(num_negatives), np.max(num_negatives)))
        logger.info("MRR@{}: {:.2f}".format(self.mrr_at_k, mean_mrr*100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mean_mrr])

        return mean_mrr