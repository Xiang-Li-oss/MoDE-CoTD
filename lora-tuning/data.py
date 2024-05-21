import os 
import json


def list_of_dicts_to_dict_of_lists(list_of_dict):
    dict_of_lists = {}
    for key in list_of_dict[0].keys():
        dict_of_lists[key] = [d[key] for d in list_of_dict]
    return dict_of_lists

def load_dataset(dataset_name):
    with open(os.path.join('dataset', '{}_train.json'.format(dataset_name)), 'r') as f:
        data_points =json.load(f)
    return data_points

def load_test_dataset(dataset_name):
    with open(os.path.join('dataset', 'testset', '{}_test.json'.format(dataset_name)), 'r') as f:
        data_points = json.load(f)
    return data_points
