# %%
import pandas as pd
import json
import re
#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %%
eval_name = "eval_make_model_aids"
eval_name = "eval_make_model_wildlife"
eval_name = "eval_make_model_bts"
#eval_name = "eval_make_model_bts_block_make"
#eval_name = "eval_make_model_ntsb"
model_name = "make_model_union"
run_name = eval_name + "_model_" + model_name
#eval_make_model_bts_block_make_model_make_model_union_predictions_all
# load JSONL
records = []
with open("../aircraft_er_predictions/" + run_name + "_predictions_all.tsv") as f:
    for line in f:
        records.append(json.loads(line))



# %%
df = pd.DataFrame(records)
print(len(df))
df.head()

# %%
df_raw = pd.read_csv("../data/ditto_aircraft/" + eval_name + "/all_pairs_with_id.txt", sep="," )
print(len(df_raw))
df_raw.head()

# %%
df["left_id"] = df_raw["left_id"] 
df["right_id"] = df_raw["right_id"] 


# %%
df.head()

# %%
### ONLY FOR BTS
df["description"] = df["right"].str.extract(
    r"COL\s+description\s+VAL\s*(.*?)(?=\n\s*COL|$)",
    expand=False
)

# %%

PAIR_RE = re.compile(r"COL\s+(?P<key>.*?)\s+VAL\s+(?P<val>.*?)(?=\s+COL\s+|$)")

def record_to_kv_list(record: str):
    """Return list like ['make: WACO', 'model: EGC', ...] in original order."""
    s = str(record)
    return [f"{m.group('key').strip()}: {m.group('val').strip()}"
            for m in PAIR_RE.finditer(s)]

# %%
# assuming your dataframe is called matches and has columns "left" and "right"
df["left_kv"]  = df["left"].apply(record_to_kv_list)
df["right_kv"] = df["right"].apply(record_to_kv_list)

# %%
df_agg = df.groupby("left_id", as_index=False).agg(
    left_count=("left", "count"),
    right_kv_distinct=("right_kv", lambda s: s.dropna().apply(tuple).nunique()),
)

# %%
df_joined = df.merge(
    df_agg,
    how="left",
    on=["left_id"],
)

# %%
df_joined

# %%
print('all evaluated potential paris: ', len(df))
df_matches = df[df["match"]==1]
print('matching paris: ', len(df_matches))

# %%
df_matches.head()

# %%
df_max = (
    df_matches
      .groupby(['right_id'], as_index=False)
      .agg({
          'match_confidence':      "max",
          'left_id':              lambda s: pd.unique(s).tolist(),
      })
      .rename(columns={'match_confidence': 'match_confidence_max', 'left_id': 'left_ids'})
      .assign(left_ids_count=lambda d: d['left_ids'].str.len())
)



# %%
df_max

# %%
df_join = df_matches.merge(
    df_max,
    left_on="right_id",
    right_on="right_id",
    how='left'
)
df_join["is_max"] = df_join["match_confidence"]==df_join["match_confidence_max"]

# %%
df_join

# %%
df_join.to_csv("../aircraft_er_predictions/" + run_name + "_all_matches.csv", index=False)

# %%
df_join_max = df_join[df_join["is_max"]==True]

df_join_max

# %%
df_join_max.to_csv('temp.csv')

df_join_max.to_csv("../aircraft_er_predictions/" + run_name + "_max_matches.csv", index=False)

# %%
df_matches[df_matches["right"].isna()]



