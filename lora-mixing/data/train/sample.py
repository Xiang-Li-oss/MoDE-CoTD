import os
import json

import random

# dataset_names = ['addsub', 'aqua', 'coin_flip', 'commonsense_qa', 'date_understanding', \
#                   'gsm8k', 'last_letter_concatenation', 'multiarith', 'single_eq', 'strategy_qa', 'svamp', 'tracking_shuffled_objects']
dataset_names = ['reclor']
sample_num = 5
for dataset_name in dataset_names:

    with open('{}_train.json'.format(dataset_name), 'r') as f:
        data_points = json.load(f)

    samples = random.sample(data_points, sample_num)

    with open('learning/{}_learn3.json'.format(dataset_name), 'w') as f:
        json.dump(samples, f, indent=2)

