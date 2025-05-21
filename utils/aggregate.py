#!/usr/bin/env python

import json
import argparse
import os
import sklearn.metrics
import pprint
import csv
from tabulate import tabulate

def calculate_metrics(ref_statements, responses):
    """
    Calculate accuracy, micro, macro and weighted f1 scores of predicted labels in responses compared to reference assessments in reference_path.
    The results are saved to output_path.

    Args:
    ref_statements (List): List of reference statements.
    responses (List): List of responses.
    """

    # get labels from responses
    pred_labels = [
        item["label"].lower()
        for ref in ref_statements 
        if (item := find_by_id(ref['id'], responses)) is not None
    ]    
    # get labels from reference
    ref_labels = [ref["assessment"].lower() for ref in ref_statements]

    # calculate metrics
    metrics = sklearn.metrics.classification_report(ref_labels, pred_labels, output_dict=True)

    return metrics

def find_by_id(id, data):
    for item in data:
        if item['id'] == id:
            return item
    return None


def aggregate_metrics(result_files: list, output_dir: str):
    """
    Takes a list of result files and computes f1 score, precision, recall and accuracy of all the results
    """
    
    y_pred = []
    y_ref = []
    no_label = []
    total_cnt = 0
    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)

            # remove items with no label
            pred_items = [item for item in data['y_pred'] if item['label'] != 'nolabel']
            ref_items = [find_by_id(item['id'], data['y_ref']) for item in pred_items]

            y_pred.extend(pred_items)
            y_ref.extend(ref_items)

            total_cnt += len(data['y_pred'])

            no_label.extend([item for item in data['y_pred'] if item['label'] == 'nolabel'])

    metrics = calculate_metrics(y_ref, y_pred)

    with open(os.path.join(output_dir, 'aggregated.json'), "w") as f:
        json.dump({
            "count": total_cnt,
            "no_label_count": len(no_label),
            "y_pred": y_pred,
            "y_ref": y_ref,
            "no_label": no_label
        }, f, indent=4, ensure_ascii=False)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=os.getcwd(), help="Path to directory with results. If not provided, defaults to current directory")
    args = parser.parse_args()

    results_dir = args.path

    result_files = [os.path.join(results_dir, filename) for filename in os.listdir(results_dir) if filename.startswith('results')]
    if not result_files:
        exit("No result files found")

    metrics_dict = aggregate_metrics(result_files, results_dir)
    table_data = []
    for (label, metrics) in metrics_dict.items():
            if label == "accuracy":
                continue

            row = [label] + [metrics.get(metric, '') for metric in ['precision', 'recall', 'f1-score', 'support']]
            table_data.append(row)

    # Display the table using tabulate
    headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    table = tabulate(table_data, headers=headers, tablefmt="grid") + "\nAccuracy: " + str(metrics_dict["accuracy"]) + "\n"
    print(table)

    # save to file
    with open("metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)

    with open("metrics.txt", "w") as f:
        f.write(table)
    with open("metrics.csv", mode="w") as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])

        # Write each class's metrics
        for label, metrics in metrics_dict.items():
            if isinstance(metrics, dict):  # Ignore 'accuracy' key, it's not a dict
                writer.writerow([
                    label,
                    round(metrics["precision"], 4),
                    round(metrics["recall"], 4),
                    round(metrics["f1-score"], 4),
                    int(metrics["support"])
                ])

        # Write overall accuracy
        writer.writerow(["accuracy", "", "", metrics_dict["accuracy"], ""])
