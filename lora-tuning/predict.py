import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig, prepare_model_for_int8_training
import argparse
import os
import json
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
from data import load_test_dataset, list_of_dicts_to_dict_of_lists

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model_id", type=str, default='google/flan-t5')
arg_parser.add_argument("--checkpoint_path", type=str)
arg_parser.add_argument("--checkpoint_name", type=str, default='all_data')
arg_parser.add_argument("--dataset_name", type=str, default='gsm8k')
arg_parser.add_argument("--batch_size", type=int, default=64)

args = arg_parser.parse_args()

checkpoint_path = args.checkpoint_path
checkpoint_name = args.checkpoint_name

peft_model_id = os.path.join(checkpoint_path, checkpoint_name)
config = PeftConfig.from_pretrained(peft_model_id)
if 'xl' in config.base_model_name_or_path:
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True)
    base_model = prepare_model_for_int8_training(base_model)
else:
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = model.to('cuda')
model.eval()

test_data_points= load_test_dataset(args.dataset_name)
test_data_points = list_of_dicts_to_dict_of_lists(test_data_points)
test_dataset = Dataset.from_dict(test_data_points)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

gen_kwargs = {
        "do_sample": True,
        "top_p": 0.7,
        "max_new_tokens": 512,
        "temperature": 0.95,
    }
predictions = []
labels = []
for batch in tqdm(test_dataloader):
   
    labels.extend(batch['output'])
    
    input_ids = tokenizer(
        batch['input'],
        padding='longest',
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )
    
    output = model.generate(input_ids=input_ids['input_ids'].cuda(), do_sample=True, top_p=0.7, max_new_tokens=512, temperature=0.95)
    prediction = tokenizer.batch_decode(output.detach().cpu().numpy(), skip_special_tokens=True)
    # batch_labels = tokenizer.batch_decode(sample['output'], skip_special_tokens=True)
    predictions.extend(prediction)


predictions = [{"prediction": p, "label": l} for p, l in zip(predictions, labels)]

if args.checkpoint_name == 'all_data':
    with open(os.path.join('baseline_lora_results', args.dataset_name+'.json'), 'w') as f:
        json.dump(predictions, f, indent=2)
else:
    result_path = 'results_xl' if 'xl' in args.model_id else 'results'
    with open(os.path.join(result_path, args.dataset_name+'.json'), 'w') as f:
        json.dump(predictions, f, indent=2)
    
