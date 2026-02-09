import csv
import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


baseline_run_names = ["baseline", "baseline_lh_0", "baseline_lh_1", "baseline_lh_2", "baseline_lh_3"]

data_name = "baseline_eval_only_canadair"
predict_dir = Path("aircraft_er_predictions")

gold_path = Path("data/ditto_aircraft") / data_name / "all_pairs.txt"

# Optional: cap how many error rows you write (helps if errors are huge)
MAX_ERROR_ROWS = None  # e.g. 200000


def parse_record_safe(record: str):
    """
    Robust Ditto record parser: extracts COL <field> VAL <value> chunks.
    Avoids splitting inside words like 'NAVAL'.
    """
    pairs = re.findall(r"(?:^|\s)COL\s+(.+?)\s+VAL\s+(.+?)(?=\s+COL\s+|$)", record.strip())
    return {k.strip(): v.strip() for k, v in pairs}


def iter_gold_labels(path):
    """Yield gold labels from all_pairs.txt line-by-line (expects gold as last TSV column)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            yield int(parts[-1])


def sklearn_from_counts(tn, fp, fn, tp, digits=3):
    """
    Get sklearn-style outputs without huge y_true/y_pred arrays.
    We create a tiny 4-row dataset and use sample_weight as the counts.
    """
    y_true = np.array([0, 0, 1, 1], dtype=np.uint8)
    y_pred = np.array([0, 1, 0, 1], dtype=np.uint8)
    w = np.array([tn, fp, fn, tp], dtype=np.int64)

    acc = accuracy_score(y_true, y_pred, sample_weight=w)
    report = classification_report(y_true, y_pred, sample_weight=w, digits=digits, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, sample_weight=w, labels=[0, 1])
    return acc, report, cm


for baseline_run_name in baseline_run_names:
    run_name = f"{data_name}_model_{baseline_run_name}"
    predict_all_path = predict_dir / f"{run_name}_predictions_all.tsv"
    print(predict_all_path)

    all_run_ts = datetime.fromtimestamp(predict_all_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    # Output paths
    metrics_path = predict_dir / f"{run_name}_eval_metrics_all.txt"
    append_path = predict_dir / "append_full_baseline_eval_metrics_all.txt"
    errors_path = predict_dir / f"{run_name}_errors_review.csv"
    aligned_path = predict_dir / f"{run_name}_aligned_errors_review.csv"

    # Running counts
    tn = fp = fn = tp = 0
    n = 0
    err_written = 0

    # Stream gold + predictions in lockstep
    gold_iter = iter_gold_labels(gold_path)

    with open(predict_all_path, "r", encoding="utf-8") as pred_f, \
         open(errors_path, "w", newline="", encoding="utf-8") as err_f, \
         open(aligned_path, "w", newline="", encoding="utf-8") as ali_f:

        err_writer = csv.DictWriter(err_f, fieldnames=[
            "left", "right", "match", "match_confidence", "gold"
        ])
        err_writer.writeheader()

        ali_writer = csv.DictWriter(ali_f, fieldnames=[
            "make_left", "make_right",
            "model_left", "model_right",
            "series_left", "series_right",
            "cert_left", "cert_right",
            "name_left", "name_right",
            "predicted", "confidence", "gold"
        ])
        ali_writer.writeheader()

        for line in pred_f:
            if not line.strip():
                continue

            pred = json.loads(line)
            try:
                gold = next(gold_iter)
            except StopIteration:
                raise RuntimeError("Gold file ended before predictions file. Are they aligned?")

            y_pred = int(pred.get("match", 0))
            y_true = int(gold)

            # update counts
            if y_true == 1 and y_pred == 1:
                tp += 1
            elif y_true == 0 and y_pred == 1:
                fp += 1
            elif y_true == 1 and y_pred == 0:
                fn += 1
            else:
                tn += 1

            # write errors incrementally
            if y_true != y_pred:
                if MAX_ERROR_ROWS is None or err_written < MAX_ERROR_ROWS:
                    err_writer.writerow({
                        "left": pred.get("left", ""),
                        "right": pred.get("right", ""),
                        "match": y_pred,
                        "match_confidence": pred.get("match_confidence", ""),
                        "gold": y_true
                    })

                    # aligned parse (only on errors)
                    left = parse_record_safe(pred.get("left", ""))
                    right = parse_record_safe(pred.get("right", ""))

                    ali_writer.writerow({
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
                        "predicted": y_pred,
                        "confidence": pred.get("match_confidence", ""),
                        "gold": y_true
                    })

                    err_written += 1

            n += 1

        # Ensure gold doesn't have extra lines
        try:
            next(gold_iter)
            raise RuntimeError("Gold file has MORE lines than predictions file. Are they aligned?")
        except StopIteration:
            pass

    # sklearn-style metrics from counts
    acc, report, cm = sklearn_from_counts(tn, fp, fn, tp, digits=3)

    # Save metrics per run
    with open(metrics_path, "w", encoding="utf-8") as f:
        print("Records:", n, file=f)
        print("Confusion counts:", {"TN": tn, "FP": fp, "FN": fn, "TP": tp}, file=f)
        print("Accuracy:", acc, file=f)
        print("\nClassification report:\n", file=f)
        print(report, file=f)
        print("\nConfusion matrix:\n", file=f)
        print(cm, file=f)

    # Append to global file
    with open(append_path, "a", encoding="utf-8") as f:
        print("Run name:", run_name, file=f)
        print("Predictions file created:", all_run_ts, "\n", file=f)
        print("Records:", n, file=f)
        print("Confusion counts:", {"TN": tn, "FP": fp, "FN": fn, "TP": tp}, file=f)
        print("Accuracy:", acc, file=f)
        print("\nClassification report:\n", file=f)
        print(report, file=f)
        print("\nConfusion matrix:\n\n", file=f)
        print(cm, file=f)
        print("\n" + "-"*60 + "\n", file=f)

    print(f"Done {run_name}: n={n}, errors_written={err_written}, cm={cm.tolist()}")
