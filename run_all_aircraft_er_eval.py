import os
import time



#special_datasets = {
#    'Structured/Beer': (32, 40)
#}

ops = """swap
swap
append_col
del
swap
drop_col
swap
swap
append_col
drop_col
drop_col
swap
del""".split('\n')


#lms = ['roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'roberta',
#       'roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'bert']

# lms = ['xlnet', 'roberta', 'roberta', 'roberta', 'xlnet', 'bert',
#        'bert', 'xlnet', 'roberta', 'bert', 'roberta', 'roberta', 'bert']

# lms = """distilbert
# bert
# xlnet
# roberta""".split('\n')

run_id = 0
datasets = """ditto_aircraft/baseline
ditto_aircraft/baseline_1
ditto_aircraft/baseline_2""".split('\n')

datasets = """aircraft_er/cictt_registry
aircraft_er/union""".split('\n')

datasets = """aircraft_er/make_model_icao_code""".split('\n')

datasets = """aircraft_er/union_make_model_2""".split('\n')
datasets = """aircraft_er/union_make_model_2""".split('\n')

datasets = """aircraft_er/baseline""".split('\n')


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

datasets = """aircraft_er/baseline
aircraft_er/baseline_lh
aircraft_er/baseline_lh_0
aircraft_er/baseline_lh_1
aircraft_er/baseline_lh_2
aircraft_er/baseline_lh_3
aircraft_er/baseline_lh_b""".split('\n')


datasets = """aircraft_er/make_model_cictt""".split('\n')

eval_dataset = "ditto_aircraft/baseline_eval_only_random_sample"
eval_dataset = "ditto_aircraft/eval_make_model_wildlife"

eval_dataset = "ditto_aircraft/make_model_bts"

lms = ['distilbert', 'distilbert']

lms = ['distilbert','distilbert','distilbert','distilbert']
lms = ['distilbert', 'distilbert','distilbert', 'distilbert','distilbert', 'distilbert', 'distilbert'] #, 'distilbert', 'distilbert', 'distilbert']
lms = ['distilbert']

for dataset, lm in zip(datasets, lms):
    print(dataset)


for dataset, lm in zip(datasets, lms):
    batch_size, max_len, epochs = 64, 64, 20

    #string variables for matcher.py
    input_path = f"data/{eval_dataset}/all_pairs.txt"
    dataset_name = dataset.rsplit("/", 1)[-1] 
    eval_dataset_name = eval_dataset.rsplit("/", 1)[-1] 
    output_path = f"aircraft_er_predictions/{eval_dataset_name}_model_{dataset_name}_predictions_all.tsv"
    #Run Matcher
    cmd = """python matcher.py \
    --task %s \
    --input_path  %s \
    --output_path  %s \
    --checkpoint_path results_ditto \
    --lm %s \
    --max_len %d \
    --use_gpu""" % (dataset, input_path, output_path, lm, max_len)
    print(cmd)
    os.system(cmd)

