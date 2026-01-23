import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# run_names is the array of the model runs to be evaluated
run_names = ["baseline", "baseline_lh","baseline_lh_0","baseline_lh_1", "baseline_lh_b", "baseline_lh_2", "baseline_lh_3"]

# global_append_filename is the name of the text file that stores metrics over all runs
global_append_filename = "append_metrics"

# predict_dir is the path of the model predictions (*_predictions_test.tsv and *_predictions_all.tsv json files)
predict_dir = Path("aircraft_er_predictions")

#Process each run_name
for run_name in run_names:

    # json filenames for the run_name
    predict_test_path = predict_dir / f"{run_name}_predictions_test.tsv"
    predict_all_path  = predict_dir / f"{run_name}_predictions_all.tsv"

    # timestamps from the prediction filenames
    test_run_ts = datetime.fromtimestamp(predict_test_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    all_run_ts = datetime.fromtimestamp(predict_all_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    # Read predictions from predictions json files - test predictions only
    records_test = []
    with open(predict_test_path) as f:
        for line in f:
            records_test.append(json.loads(line))

    # Read predictions from predictions json files - all predictions
    records = []
    with open(predict_all_path) as f:
        for line in f:
            records.append(json.loads(line))

    # Create pandas dataframes with predictions
    df_all = pd.DataFrame(records)
    df_test = pd.DataFrame(records_test)

    # Read original ground truth data for test and all_pairs
    gold_test = pd.read_csv("data/ditto_aircraft/" + run_name + "/test.txt", sep="\t", header=None, names=["left", "right", "gold"])
    gold_all = pd.read_csv("data/ditto_aircraft/" + run_name + "/all_pairs.txt", sep="\t", header=None, names=["left", "right", "gold"])

    # Add ground truth column ("gold") to predicitons dataframes
    df_all["gold"] = gold_all["gold"]
    df_test["gold"] = gold_test["gold"]

    # Create y_true and y_pred for scikit learn accuracy score, classification report, 
    # and confusion matrix printouts (test set only)
    y_true_test = df_test["gold"]
    y_pred_test = df_test["match"]

    # Save metrics to individual file for each run
    with open("aircraft_er_predictions/" + run_name + "_metrics_test.txt", "w") as f:
        print("Accuracy:", accuracy_score(y_true_test, y_pred_test), file=f)
        print("\nClassification report:\n", file=f)
        print(classification_report(y_true_test, y_pred_test), file=f)
        print("\nConfusion matrix:\n", file=f)
        print(confusion_matrix(y_true_test, y_pred_test), file=f)

    # Append run metrics to global file
    with open("aircraft_er_predictions/", global_append_filename, "_test.txt", "a") as f:
        print("\nRun name: ", run_name, file=f)
        print("Predictions file created:", test_run_ts, "\n", file=f)
        print("Accuracy:", accuracy_score(y_true_test, y_pred_test), file=f)
        print("\nClassification report:\n", file=f)
        print(classification_report(y_true_test, y_pred_test), file=f)
        print("\nConfusion matrix:\n\n", file=f)
        print(confusion_matrix(y_true_test, y_pred_test), file=f)    

    # Create y_true and y_pred for scikit learn accuracy score, classification report, 
    # and confusion matrix printouts (all_pairs reference)
    y_true = df_all["gold"]
    y_pred = df_all["match"]

    # Save metrics to individual file for each run
    with open("aircraft_er_predictions/" + run_name + "_metrics_all.txt", "w") as f:
        print("Accuracy:", accuracy_score(y_true, y_pred), file=f)
        print("\nClassification report:\n", file=f)
        print(classification_report(y_true, y_pred), file=f)
        print("\nConfusion matrix:\n", file=f)
        print(confusion_matrix(y_true, y_pred), file=f)
    
    # Save metrics to individual file for each run
    with open("aircraft_er_predictions/", global_append_filename, "_all.txt", "a") as f:
        print("Run name: ", run_name, file=f)
        print("Predictions file created: ", all_run_ts, "\n", file=f)
        print("Accuracy:", accuracy_score(y_true, y_pred), file=f)
        print("\nClassification report:\n", file=f)
        print(classification_report(y_true, y_pred), file=f)
        print("\nConfusion matrix:\n\n", file=f)
        print(confusion_matrix(y_true, y_pred), file=f)    

    # Save errors for test dataset
    errors = df_test[df_test["gold"] != df_test["match"]]
    errors.to_csv("aircraft_er_predictions/" + run_name + "_errors_review.csv", index=False)

    # Parse the record to get raw fields from ditto string
    def parse_record(record: str):
        """Parse Ditto serialized record into a dict of {field: value}."""
        parts = re.split(r"COL |VAL ", record.strip())
        parts = [p for p in parts if p]  # drop empties
        return {parts[i].strip(): parts[i+1].strip() for i in range(0, len(parts), 2)}

    # Create an errors file with original parsed field names
    # format specific to baseline model (cictt-faa make, model, series, tc, name)
    parsed = []
    for _, row in errors.iterrows():
        left = parse_record(row["left"])
        right = parse_record(row["right"])
        parsed.append({
            "cictt_make": left.get("make"),
            "make": right.get("make"),
            "cictt_model": left.get("model"),
            "model": right.get("model"),
            "cictt_series": left.get("series"),
            "series": right.get("series"),
            "cictt_cert": left.get("cert"),
            "cert": right.get("cert"),
            "cictt_name": left.get("name"),
            "name": right.get("name"),
            "predicted": row["match"],
            "confidence": row["match_confidence"],
            "gold": row["gold"]
        })

    # Save aligned parsed data to dataframe and write to file
    aligned = pd.DataFrame(parsed)
    aligned.to_csv("aircraft_er_predictions/" + run_name + "_aligned_errors_review.csv", index=False)