import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# run_names is the array of the model runs to be evaluated
baseline_run_names = ["baseline", "baseline_lh","baseline_lh_0","baseline_lh_1", "baseline_lh_b", "baseline_lh_2", "baseline_lh_3"]

# data_name is the evaluation dataset (eval data_name dataset against run_name model.pt)
####data_name = "baseline_eval_only_random_sample"
data_name = "baseline_eval_only_canadair"

# predict_dir is the path of the model predictions (*_predictions_test.tsv and *_predictions_all.tsv json files)
predict_dir = Path("aircraft_er_predictions")

#Process each run_name
for baseline_run_name in baseline_run_names:

    # json filenames for the run_name
    run_name = data_name + "_model_" + baseline_run_name
    predict_all_path  = predict_dir / f"{run_name}_predictions_all.tsv"
    print(predict_all_path)
    
    # timestamps from the prediction filename
    all_run_ts = datetime.fromtimestamp(predict_all_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    # Read predictions from predictions json files - all predictions (no test set for eval only)
    records = []
    with open(predict_all_path) as f:
        for line in f:
            records.append(json.loads(line))

    # Create pandas dataframes with predictions
    df_all = pd.DataFrame(records)
    print('records: ', len(df_all))

    # Read original ground truth data for test and all_pairs
    gold_all = pd.read_csv("data/ditto_aircraft/" + data_name + "/all_pairs.txt", sep="\t", header=None, names=["left", "right", "gold"])
    print('gold: ',len(gold_all))

    # Add ground truth column ("gold") to predicitons dataframes
    df_all["gold"] = gold_all["gold"]

    # Create y_true and y_pred for scikit learn accuracy score, classification report, 
    # and confusion matrix printouts 
    y_true = df_all["gold"]
    y_pred = df_all["match"]

    # Save metrics to individual file for each run
    with open("aircraft_er_predictions/" + run_name + "_eval_metrics_all.txt", "w") as f:
        print("Accuracy:", accuracy_score(y_true, y_pred), file=f)
        print("\nClassification report:\n", file=f)
        print(classification_report(y_true, y_pred), file=f)
        print("\nConfusion matrix:\n", file=f)
        print(confusion_matrix(y_true, y_pred), file=f)

    # Append run metrics to global file
    with open("aircraft_er_predictions/append_baseline_eval_metrics_all.txt", "a") as f:
        print("Run name: ", run_name, file=f)
        print("Predictions file created: ", all_run_ts, "\n", file=f)
        print("Accuracy:", accuracy_score(y_true, y_pred), file=f)
        print("\nClassification report:\n", file=f)
        print(classification_report(y_true, y_pred), file=f)
        print("\nConfusion matrix:\n\n", file=f)
        print(confusion_matrix(y_true, y_pred), file=f)    

    # Save errors for all_pairs dataset
    errors = df_all[df_all["gold"] != df_all["match"]]
    errors.to_csv("aircraft_er_predictions/" + run_name + "_errors_review.csv", index=False)

    # Parse the record to get raw fields from ditto string
    def parse_record(record: str):
        """Parse Ditto serialized record into a dict of {field: value}."""
        parts = re.split(r"COL |VAL ", record.strip())
        parts = [p for p in parts if p]  # drop empties
        return {parts[i].strip(): parts[i+1].strip() for i in range(0, len(parts), 2)}


    # This portion needs to be customized for the left and right files 
    parsed = []
    for _, row in errors.iterrows():
        left = parse_record(row["left"])
        right = parse_record(row["right"])
        parsed.append({
            "make_left": left.get("make"),
            "make_right": right.get("make"),
            "model_left": left.get("model"),
            "model_right": right.get("model"),
            "series_left": left.get("series"),
            "series_right": right.get("series"),
            "cert_left": left.get("cert"),
            "cert_right": right.get("cert"),
            "name_left": left.get("name"),
            "name_right": right.get("name"),
            "predicted": row["match"],
            "confidence": row["match_confidence"],
            "gold": row["gold"]
        })
    
    # Save aligned parsed data to dataframe and write to file
    aligned = pd.DataFrame(parsed)
    aligned.to_csv("aircraft_er_predictions/" + run_name + "_aligned_errors_review.csv", index=False)