# %%
import pandas as pd
import json
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %%
run_name = "baseline"
run_name = "baseline_lh"
run_name = "baseline_lh_2"
run_name = "baseline_lh_3"

run_name = "baseline_lh_1"
run_name = "baseline_lh_0"
run_name = "baseline"

# load JSONL
records = []
with open("../aircraft_er_predictions/" + run_name + "_predictions_all.tsv") as f:
    for line in f:
        records.append(json.loads(line))


        # load JSONL
records_test = []
with open("../aircraft_er_predictions/" + run_name + "_predictions_test.tsv") as f:
    for line in f:
        records_test.append(json.loads(line))

# %%
df_all = pd.DataFrame(records)
df_test = pd.DataFrame(records_test)
df_test.head()

# %%
gold_test = pd.read_csv("../data/ditto_aircraft/" + run_name + "/test.txt", sep="\t", header=None, names=["left", "right", "gold"])

gold_all = pd.read_csv("../data/ditto_aircraft/" + run_name + "/all_pairs.txt", sep="\t", header=None, names=["left", "right", "gold"])
print(gold_all.head())

# %%
df_all["gold"] = gold_all["gold"]
df_test["gold"] = gold_test["gold"]

# %%
#Test Data

y_true_test = df_test["gold"]
y_pred_test = df_test["match"]

print("Accuracy:", accuracy_score(y_true_test, y_pred_test))
print("\nClassification report:\n", classification_report(y_true_test, y_pred_test,digits=3))
print("\nConfusion matrix:\n", confusion_matrix(y_true_test, y_pred_test))



with open("../aircraft_er_predictions/" + run_name + "_metrics_test.txt", "w") as f:
    print("Accuracy:", accuracy_score(y_true_test, y_pred_test), file=f)
    print("\nClassification report:\n", file=f)
    print(classification_report(y_true_test, y_pred_test,digits=3), file=f)
    print("\nConfusion matrix:\n", file=f)
    print(confusion_matrix(y_true_test, y_pred_test), file=f)

# %%
#All Data

y_true = df_all["gold"]
y_pred = df_all["match"]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification report:\n", classification_report(y_true, y_pred,digits=3))
print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))



with open("../aircraft_er_predictions/" + run_name + "_metrics_all.txt", "w") as f:
    print("Accuracy:", accuracy_score(y_true, y_pred), file=f)
    print("\nClassification report:\n", file=f)
    print(classification_report(y_true, y_pred, digits=3), file=f)
    print("\nConfusion matrix:\n", file=f)
    print(confusion_matrix(y_true, y_pred), file=f)

# %%
errors = df_test[df_test["gold"] != df_test["match"]]
print(errors[["left","right","gold","match","match_confidence"]].head(20))

# %%
errors

# %%
errors.to_csv("../aircraft_er_predictions/" + run_name + "_errors_review.csv", index=False)

# %%
def parse_record(record: str):
    """Parse Ditto serialized record into a dict of {field: value}."""
    parts = re.split(r"COL |VAL ", record.strip())
    parts = [p for p in parts if p]  # drop empties
    return {parts[i].strip(): parts[i+1].strip() for i in range(0, len(parts), 2)}

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

aligned = pd.DataFrame(parsed)

# %%
aligned.to_csv("../aircraft_er_predictions/" + run_name + "_aligned_errors_review.csv", index=False)

# %%
aligned.head()

# %%

pairs_with_id_path = f"../data/ditto_aircraft/{run_name}/all_pairs_with_id.txt"
pairs_with_id = pd.read_csv(pairs_with_id_path)


# %%
pairs_with_id

# %%
df_all

# %%
df_all["TP"] = ((df_all["match"] == 1) & (df_all["gold"] == 1)).astype(int)
df_all["TN"] = ((df_all["match"] == 0) & (df_all["gold"] == 0)).astype(int)
df_all["FP"] = ((df_all["match"] == 1) & (df_all["gold"] == 0)).astype(int)
df_all["FN"] = ((df_all["match"] == 0) & (df_all["gold"] == 1)).astype(int)

# %%
df_all.head()

# %%


df_agg_cictt = (
      df_all.groupby('left_id', as_index=False)
      .agg({'left':         'first',                          
            'TP':           'sum',
            'TN':           'sum',
            'FP':           'sum',
            'FN':           'sum',                    
            'right':        lambda s: pd.unique(s.dropna()).tolist(),

      })
)
df_agg_cictt["right_len"] = df_agg_cictt["right"].apply(len)

df_agg_cictt.head()

# %%
df_agg_max_conf = (
      df_all[df_all["match"]==1].groupby('left_id', as_index=False)
      .agg({'match_confidence':         'max',                          
      })
)
df_agg_max_conf["max_conf_match"] = df_agg_max_conf["match_confidence"]

df_agg_max_conf

# %%
df_all_merged = df_all.merge(
    df_agg_cictt,
    on="left_id",
    how="left",
    suffixes=("", "_agg"))

# %%
df_all_merged = df_all_merged.merge(
    df_agg_max_conf,
    on="left_id",
    how="left",
    suffixes=("", "_max"))

# %%
df_all_merged 

# %%

df_all_merged["best"] = ((df_all_merged["match"]==1) & (df_all_merged["match_confidence"] == df_all_merged["match_confidence_max"])).astype(int)

# %%
df_all_merged

# %%
df_all_merged[df_all_merged["best"]==1]


