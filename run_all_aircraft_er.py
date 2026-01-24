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
datasets = """aircraft_er/baseline
aircraft_er/baseline_1
aircraft_er/baseline_2""".split('\n')

datasets = """aircraft_er/cictt_registry
aircraft_er/union""".split('\n')

datasets = """aircraft_er/make_model_icao_code""".split('\n')

datasets = """aircraft_er/union_make_model_2""".split('\n')


datasets = """aircraft_er/baseline_lh""".split('\n')

datasets = """aircraft_er/baseline_lh_2""".split('\n')
datasets = """aircraft_er/baseline_lh_3""".split('\n')

datasets = """aircraft_er/baseline
aircraft_er/baseline_lh
aircraft_er/baseline_lh_0
aircraft_er/baseline_lh_1
aircraft_er/baseline_lh_2
aircraft_er/baseline_lh_3""".split('\n')


datasets = """aircraft_er/baseline_lh""".split('\n')

datasets = """aircraft_er/baseline""".split('\n')

lms = ['distilbert', 'distilbert', 'distilbert', 'distilbert']


lms = ['distilbert']

lms = ['distilbert', 'distilbert','distilbert', 'distilbert','distilbert', 'distilbert']

for dataset, lm in zip(datasets, lms):
    print(dataset)


#for dataset, op, lm in zip(datasets, ops, lms):
#    if dataset in special_datasets:
#        batch_size, epochs = special_datasets[dataset]
#    else:



for dataset, lm in zip(datasets, lms):
    batch_size, max_len, epochs = 64, 64, 20

    #for da in [True, False]:
    #    for dk in [True, False]:
    #        for run_id in range(5):
    
            #--run_id %d""" % (dataset, batch_size, lm, epochs, run_id)
    print(dataset)
    ##Run DITTO
    cmd = """CUDA_VISIBLE_DEVICES=0 python train_ditto.py \
    --task %s \
    --logdir results_ditto/ \
    --batch_size %d \
    --max_len %d \
    --finetuning \
    --lr 3e-5 \
    --fp16 \
    --lm %s \
    --n_epochs %d \
    --run_id %s \
    --save_model""" % (dataset, batch_size, max_len, lm, epochs, run_id)
    print(cmd)
   #os.system(cmd)


    #string variables for matcher.py
    dataset_name = dataset.rsplit("/", 1)[-1] 
    input_path = f"data/ditto_aircraft/{dataset_name}/all_pairs.txt"
    output_path = f"aircraft_er_predictions/{dataset_name}_predictions_all.tsv"
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

    #string variables for matcher.py
    dataset_name = dataset.rsplit("/", 1)[-1] 
    input_path = f"data/ditto_aircraft/{dataset_name}/test.txt"
    output_path = f"aircraft_er_predictions/{dataset_name}_predictions_test.tsv"
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