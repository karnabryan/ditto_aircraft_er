import os
import time
import csv
import subprocess
from datetime import datetime


run_id = 0
datasets = """aircraft_er/baseline
aircraft_er/baseline_1
aircraft_er/baseline_2""".split('\n')

datasets = """aircraft_er/cictt_registry
aircraft_er/union""".split('\n')

datasets = """aircraft_er/cictt_registry
aircraft_er/faa_registry""".split('\n')

datasets = """aircraft_er/union""".split('\n')


datasets = """aircraft_er/baseline
aircraft_er/baseline_lh
aircraft_er/baseline_lh_0
aircraft_er/baseline_lh_1
aircraft_er/baseline_lh_2
aircraft_er/baseline_lh_3
aircraft_er/baseline_lh_b
aircraft_er/cictt_registry
aircraft_er/faa_registry
aircraft_er/union""".split('\n')


datasets = """aircraft_er/make_model_reg""".split('\n')

datasets = """aircraft_er/make_model_cictt
aircraft_er/make_model_cictt_no_lh
aircraft_er/make_model_faa
aircraft_er/make_model_faa_no_lh
aircraft_er/make_model_reg
aircraft_er/make_model_reg_no_lh
aircraft_er/make_model_doc8643
aircraft_er/make_model_doc8643_no_lh
aircraft_er/make_model_doc8643_with_drops
aircraft_er/make_model_doc8643_with_drops_no_lh
aircraft_er/make_model_doc8643_code
aircraft_er/make_model_doc8643_description""".split('\n')

datasets = """aircraft_er/make_model_doc8643_code
aircraft_er/make_model_doc8643_description""".split('\n')

lms = ['distilbert', 'distilbert', 'distilbert', 'distilbert']


lms = ['distilbert']


lms = ['distilbert', 'distilbert','distilbert', 'distilbert','distilbert', 'distilbert', 'distilbert', 'distilbert', 'distilbert', 'distilbert','distilbert', 'distilbert']
lms = ['distilbert', 'distilbert']
log_path = "run_timings.csv"
file_exists = os.path.exists(log_path)

def run_timed(cmd, name, extra=None, env=None):
    """
    cmd: list[str] or str (if shell=True). Use list[str].
    name: label like "train", "match_all", "match_test"
    extra: dict of extra fields to log
    """
    extra = extra or {}
    start_ts = time.time()
    start_iso = datetime.now().isoformat(timespec="seconds")

    print(f"\n[{start_iso}] START {name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, env=env)

    end_ts = time.time()
    end_iso = datetime.now().isoformat(timespec="seconds")
    elapsed = end_ts - start_ts

    print(f"[{end_iso}] END   {name} (rc={result.returncode}) in {elapsed:.2f}s")

    row = {
        "start_iso": start_iso,
        "end_iso": end_iso,
        "elapsed_sec": round(elapsed, 3),
        "step": name,
        "returncode": result.returncode,
        **extra,
        "cmd": " ".join(cmd),
    }

    # append to CSV
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists and f.tell() == 0:
            w.writeheader()
        w.writerow(row)

    return result.returncode, elapsed


for dataset, lm in zip(datasets, lms):
    batch_size, max_len, epochs = 64, 64, 40
    run_id = 0  # set however you want

    dataset_name = dataset.rsplit("/", 1)[-1]

    # ----- TRAIN -----
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    train_cmd = [
        "python", "train_ditto.py",
        "--task", dataset,
        "--logdir", "results_ditto/",
        "--batch_size", str(batch_size),
        "--max_len", str(max_len),
        "--finetuning",
        "--lr", "3e-5",
        "--fp16",
        "--lm", lm,
        "--n_epochs", str(epochs),
        "--run_id", str(run_id),
        "--save_model",
    ]

    rc, _ = run_timed(
        train_cmd,
        "train",
        extra={"dataset": dataset, "lm": lm, "run_id": run_id},
        env=env,
    )
    if rc != 0:
        print("Training failed; skipping matcher for this dataset.")
        continue

    # ----- MATCH (all_pairs) -----
    input_path = f"data/ditto_aircraft/{dataset_name}/all_pairs.txt"
    output_path = f"aircraft_er_predictions/{dataset_name}_predictions_all.tsv"

    match_all_cmd = [
        "python", "matcher.py",
        "--task", dataset,
        "--input_path", input_path,
        "--output_path", output_path,
        "--checkpoint_path", "results_ditto",
        "--lm", lm,
        "--max_len", str(max_len),
        "--use_gpu",
    ]
    run_timed(match_all_cmd, "match_all", extra={"dataset": dataset, "lm": lm, "run_id": run_id})

    # ----- MATCH (test) -----
    input_path = f"data/ditto_aircraft/{dataset_name}/test.txt"
    output_path = f"aircraft_er_predictions/{dataset_name}_predictions_test.tsv"

    match_test_cmd = [
        "python", "matcher.py",
        "--task", dataset,
        "--input_path", input_path,
        "--output_path", output_path,
        "--checkpoint_path", "results_ditto",
        "--lm", lm,
        "--max_len", str(max_len),
        "--use_gpu",
    ]
    run_timed(match_test_cmd, "match_test", extra={"dataset": dataset, "lm": lm, "run_id": run_id})