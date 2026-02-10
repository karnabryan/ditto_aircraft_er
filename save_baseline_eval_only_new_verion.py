import csv
import json
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


baseline_run_names = ["baseline", "baseline_lh_0", "baseline_lh_1", "baseline_lh_2", "baseline_lh_3"]

data_name = "baseline_eval_only_exhaustive_new"

gold_path = Path("data/ditto_aircraft") / data_name / "all_pairs.txt"
# IDs aligned with gold_path (same order/rows as all_pairs.txt)
ids_path = Path("data/ditto_aircraft") / data_name / "all_pairs_with_id.txt"
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

def iter_ids(path):
    """
    Yield (left_id, right_id) from an all_pairs_with_id file aligned line-for-line.
    Adjust column names if needed.
    """
    df = pd.read_csv(path, usecols=["left_id", "right_id"], dtype=str)
    for row in df.itertuples(index=False):
        yield row.left_id, row.right_id

def as_p_match(match, match_conf):
    """
    Convert Ditto outputs to estimated P(match==1).
    Ditto's match_confidence is confidence in the predicted class.
    """
    mc = float(match_conf)
    m = int(match)
    return mc if m == 1 else (1.0 - mc)

for baseline_run_name in baseline_run_names:
    run_name = f"{data_name}_model_{baseline_run_name}"
    predict_all_path = Path(f"{run_name}_predictions_all.tsv")
    predict_dir = Path("aircraft_er_predictions")
    predict_all_path = predict_dir / f"{run_name}_predictions_all.tsv"
    print(predict_all_path)

    all_run_ts = datetime.fromtimestamp(predict_all_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    # Output paths
    metrics_path = f"{run_name}_eval_metrics_all.txt"
    append_path = "append_full_baseline_eval_metrics_all.txt"
    errors_path = f"{run_name}_errors_review.csv"
    aligned_path = f"{run_name}_aligned_errors_review.csv"
    # Output paths 
    metrics_path = predict_dir / f"{run_name}_eval_metrics_all.txt" 
    append_path = predict_dir / "append_full_baseline_eval_metrics_all.txt" 
    errors_path = predict_dir / f"{run_name}_errors_review.csv" 
    aligned_path = predict_dir / f"{run_name}_aligned_errors_review.csv"   

    # Running counts
    tn = fp = fn = tp = 0
    n = 0
    err_written = 0
    
    # --- TOP-1 accumulators (per right_id) ---
    # For each right_id, keep the best-scoring candidate (highest p_match)
    best_for_right = {}  # right_id -> dict(score, gold, left, right, match, conf)

    # Track the gold true link existence per right_id (should be 1 for most setups)
    gold_pos_seen = set()  # right_ids that have at least one gold==1 pair

    # Stream gold + predictions in lockstep
    gold_iter = iter_gold_labels(gold_path)
    ids_iter = iter_ids(ids_path)

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
            y_true = int(gold)   # <-- ADD THIS LINE
            
            # ids aligned with this pair
            left_id, right_id = next(ids_iter)

            # track whether this right_id has a gold positive (for FN accounting)
            if y_true == 1:
                gold_pos_seen.add(right_id)

            # --- TOP-1 update ---
            # Mode A: use p_match ranking (recommended for "top-1")
            score = as_p_match(y_pred, pred.get("match_confidence", 0.0))

            # Mode B (if you prefer your original logic):
            # only consider predicted matches; score = match_confidence; skip match==0
            # if y_pred == 0:
            #     score = None
            # else:
            #     score = float(pred.get("match_confidence", 0.0))

            if score is not None:
                cur = best_for_right.get(right_id)
                if (cur is None) or (score > cur["score"]):
                    best_for_right[right_id] = {
                        "score": score,
                        "gold": y_true,
                        "left_id": left_id,
                        "right_id": right_id,
                        "left": pred.get("left", ""),
                        "right": pred.get("right", ""),
                        "match": y_pred,
                        "match_confidence": pred.get("match_confidence", "")
                    }


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
        try:
            next(ids_iter)
            raise RuntimeError("IDs file has MORE lines than predictions file. Are they aligned?")
        except StopIteration:
            pass



    # sklearn-style metrics from counts
    acc, report, cm = sklearn_from_counts(tn, fp, fn, tp, digits=3)

    # --- TOP-1 metrics (per right_id) ---
    # Predicted links = one per right_id where we have a best candidate (depending on mode)
    n_right_true = len(gold_pos_seen)  # right_ids that truly have a match in this evaluation
    n_right_pred = len(best_for_right)

    top1_tp = sum(1 for r in best_for_right.values() if r["gold"] == 1)
    top1_fp = n_right_pred - top1_tp
    top1_fn = n_right_true - top1_tp  # missed true rights (includes abstentions if mode B)

    top1_precision = top1_tp / (top1_tp + top1_fp) if (top1_tp + top1_fp) else 0.0
    top1_recall = top1_tp / (top1_tp + top1_fn) if (top1_tp + top1_fn) else 0.0
    top1_f1 = (2 * top1_precision * top1_recall / (top1_precision + top1_recall)) if (top1_precision + top1_recall) else 0.0
    top1_coverage = n_right_pred / n_right_true if n_right_true else 0.0



    # Save metrics per run
    with open(metrics_path, "w", encoding="utf-8") as f:
        print("Records:", n, file=f)
        print("Confusion counts:", {"TN": tn, "FP": fp, "FN": fn, "TP": tp}, file=f)
        print("Accuracy:", acc, file=f)
        print("\nClassification report:\n", file=f)
        print(report, file=f)
        print("\nConfusion matrix:\n", file=f)
        print(cm, file=f)
        print("\nTOP-1 (per right_id) metrics:", file=f)
        print({
            "n_right_true": n_right_true,
            "n_right_pred": n_right_pred,
            "coverage": round(top1_coverage, 6),
            "TP": top1_tp,
            "FP": top1_fp,
            "FN": top1_fn,
            "precision": round(top1_precision, 6),
            "recall": round(top1_recall, 6),
            "f1": round(top1_f1, 6),
        }, file=f)


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
        print("\nTOP-1 (per right_id) metrics:", file=f)
        print({
            "n_right_true": n_right_true,
            "n_right_pred": n_right_pred,
            "coverage": round(top1_coverage, 6),
            "TP": top1_tp,
            "FP": top1_fp,
            "FN": top1_fn,
            "precision": round(top1_precision, 6),
            "recall": round(top1_recall, 6),
            "f1": round(top1_f1, 6),
        }, file=f)


    print(f"Done {run_name}: n={n}, errors_written={err_written}, cm={cm.tolist()}")
