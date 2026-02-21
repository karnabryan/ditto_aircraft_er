# save_top1_samples.py
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


eval_names = ["eval_make_model_aids","eval_make_model_wildlife","eval_make_model_bts","eval_make_model_ntsb",
"eval_make_model_bts_block_make"]
eval_names= ["make_model_bts"]
model_name = "make_model_bts"
eval_names= ["baseline_1_exhaustive"]
model_name = "baseline_1"
pred_dir = Path("../aircraft_er_predictions")
data_dir = Path("../data/ditto_aircraft")

SAMPLE_N = 100
RANDOM_STATE = 99

# If True: only consider rows where Ditto predicted match==1 for top1 selection
# If False (recommended): rank by p_match so match==0 w/ high confidence doesn't dominate incorrectly
ONLY_PREDICTED_MATCHES = False


PAIR_RE = re.compile(r"(?:^|\s)COL\s+(.+?)\s+VAL\s+(.+?)(?=\s+COL\s+|$)")

def parse_ditto_record(record: str) -> dict:
    """Parse 'COL k VAL v' into {k:v} safely."""
    pairs = PAIR_RE.findall(str(record).strip())
    return {k.strip(): v.strip() for k, v in pairs}

def load_jsonl_predictions(path: Path) -> pd.DataFrame:
    """Each line is JSON (even if file extension is .tsv)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)

def load_ids_aligned(path: Path) -> pd.DataFrame:
    """
    Load all_pairs_with_id.txt which you said is comma-separated.
    Must contain left_id, right_id in the same row order as predictions/all_pairs.
    """
    df = pd.read_csv(path, sep=",", dtype=str)
    # normalize column names just in case
    cols = {c.strip(): c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    if "left_id" not in df.columns or "right_id" not in df.columns:
        raise ValueError(f"Expected left_id/right_id in {path}. Found columns: {df.columns.tolist()}")

    return df[["left_id", "right_id"]].copy()

def add_match_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add p_match score for ranking.
    Ditto's match_confidence is confidence in the predicted class.
    """
    out = df.copy()
    out["match"] = pd.to_numeric(out.get("match", 0), errors="coerce").fillna(0).astype(int)
    out["match_confidence"] = pd.to_numeric(out.get("match_confidence", 0.0), errors="coerce").fillna(0.0)

    out["p_match"] = np.where(
        out["match"] == 1,
        out["match_confidence"],
        1.0 - out["match_confidence"],
    )
    return out

def top1_per_left_match1(df: pd.DataFrame, score_col="match_confidence") -> pd.DataFrame:
    df_m = df[df["match"] == 1].copy()
    if df_m.empty:
        return df_m  # nothing predicted as match

    idx = df_m.groupby("left_id")[score_col].idxmax()
    top1 = df_m.loc[idx].copy()
    top1.sort_values([score_col], ascending=False, inplace=True)
    return top1

def top1_per_right_match1(df: pd.DataFrame, score_col="match_confidence") -> pd.DataFrame:
    df_m = df[df["match"] == 1].copy()
    if df_m.empty:
        return df_m

    idx = df_m.groupby("right_id")[score_col].idxmax()
    top1 = df_m.loc[idx].copy()
    top1.sort_values([score_col], ascending=False, inplace=True)
    return top1

### MAIN LOOP ###
for eval_name in eval_names:
    run_name = f"{eval_name}_model_{model_name}"
    #run_name = eval_name
    pred_path = pred_dir / f"{run_name}_predictions_test.tsv"

    ids_path = data_dir / eval_name / "test_with_id.txt"

    print(f"\n=== {run_name} ===")
    print("Pred:", pred_path)
    print("IDs :", ids_path)

    df_pred = load_jsonl_predictions(pred_path)
    df_ids = load_ids_aligned(ids_path)

    # Align by row order (fast + reliable if files were produced in lockstep)
    df_pred = df_pred.reset_index(drop=True)
    df_ids = df_ids.reset_index(drop=True)

    if len(df_pred) != len(df_ids):
        raise RuntimeError(
            f"Length mismatch for {eval_name}: preds={len(df_pred)} ids={len(df_ids)}. "
            "These must be aligned line-for-line."
        )

    df = df_pred.copy()
    df["left_id"] = df_ids["left_id"].values
    df["right_id"] = df_ids["right_id"].values

    # Add ranking score
    df = add_match_score(df)

    # Optional: keep only predicted matches before selecting top1
    if ONLY_PREDICTED_MATCHES:
        df_rank = df[df["match"] == 1].copy()
    else:
        df_rank = df

    # If some left_ids have no rows after filtering, top1 will drop them.
    # That’s fine for manual review, but we’ll report coverage.
    n_left_total = df["left_id"].nunique()
    n_left_rank  = df_rank["left_id"].nunique()

    # Compute top1 per fixed left_id
    df_top1 = top1_per_right_match1(df, score_col="match_confidence")
    df_top1 = top1_per_left_match1(df, score_col="match_confidence")

    # Add a few parsed fields to make manual review easier
    # (works even if fields differ across datasets)
    left_parsed = df_top1["left"].apply(parse_ditto_record) if "left" in df_top1.columns else pd.Series([{}]*len(df_top1))
    right_parsed = df_top1["right"].apply(parse_ditto_record) if "right" in df_top1.columns else pd.Series([{}]*len(df_top1))

    left_df = pd.json_normalize(left_parsed).add_prefix("L_")
    right_df = pd.json_normalize(right_parsed).add_prefix("R_")
    df_top1 = pd.concat([df_top1.reset_index(drop=True), left_df, right_df], axis=1)

    # Save full top1 file for this eval set
    out_top1 = pred_dir / f"{run_name}_top1_left.csv"
    df_top1.to_csv(out_top1, index=False)

    # Random sample for manual labeling (from top1 only)
    sample_n = min(SAMPLE_N, len(df_top1))
    df_sample = df_top1.sample(n=sample_n, random_state=RANDOM_STATE) if sample_n > 0 else df_top1.copy()

    # Add empty label column for your manual decision
    df_sample = df_sample.copy()
    df_sample["manual_label"] = ""  # fill with 1/0 after review
    df_sample["notes"] = ""

    out_sample = pred_dir / f"{run_name}_top1_left_sample_{sample_n}.csv"
    df_sample.to_csv(out_sample, index=False)

    # Simple summary
    print(f"pairs: {len(df):,}")
    print(f"unique left_id: {n_left_total:,}")
    print(f"left_id covered in ranking set: {n_left_rank:,} ({n_left_rank/n_left_total:.3f})")
    print(f"top1 rows: {len(df_top1):,}")
    print(f"saved: {out_top1}")
    print(f"saved sample: {out_sample}")

    # Optional: quick score distribution
    print("p_match quantiles:", df_top1["p_match"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict())
