import os
import json
import random
import jsonlines


def get_examples_for_learning(dataset_name):
    file_path = 'data/learn/{}.jsonl'.format(dataset_name)
    samples = list(jsonlines.open(file_path))
    return samples

def get_examples_for_inference(dataset_name):
    
    file_path = 'data/evaluation/{}_test.json'.format(dataset_name)
    with open(file_path) as f:
        data_points = json.load(f)
    return data_points