import os
import json
import re
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dataset_name', type=str)
args = arg_parser.parse_args()

def extract_answer(dataset_name, prediction):
    if '-->' in prediction:
        a = prediction.split('-->')[-1]
    else:
        if dataset_name in ['addsub', 'gsm8k', 'multiarith', 'single_eq', 'svamp']:
            candidates = [s for s in re.findall(r'-?\d+\.?\d*', prediction)]
            a = candidates[-1] if len(candidates) > 0 else 'None'
        
        elif dataset_name in ['openbook_qa', 'reclor', 'aqua', 'commonsense_qa', 'date_understanding', 'last_letter_concatenation', 'tracking_shuffled_objects']:
            candidates = [s for s in re.findall(r'A|B|C|D|E|F', prediction)]
            a = candidates[-1] if len(candidates) > 0 else 'None'
        elif dataset_name in ['coin_flip' , 'strategy_qa']:
            candidates = [s for s in re.findall(r'Yes|No|yes|no|true|false|True|False', prediction)]
            a = candidates[-1] if len(candidates) > 0 else 'None'
    if dataset_name in ['coin_flip', 'strategy_qa']:
        ans = 'yes' if 'true' in a.lower() or 'yes' in a.lower() else 'no'
        a = ans
    return a

with open('predictions/{}.json'.format(args.dataset_name, 'r')) as f:
    results = json.load(f)

correct = 0
for result in results:
    g, p = result['label'], result['prediction']
    if g.lower() in extract_answer(args.dataset_name, p).lower():
        correct += 1

print('{:.4f}'.format(correct / len(results)))
