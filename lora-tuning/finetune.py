
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_int8_training
import os
from data import load_dataset, list_of_dicts_to_dict_of_lists
from datasets import Dataset
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--model_id', type=str, default='google/flan-t5-large')
arg_parser.add_argument('--dataset_name', type=str)
arg_parser.add_argument('--epoch', type=int, default=5)
arg_parser.add_argument('--batch_size', type=int, default=16)
arg_parser.add_argument('--log_path', type=str)
arg_parser.add_argument('--save_path', type=str)

args = arg_parser.parse_args()

dataset_name = args.dataset_name
model_id = args.model_id
tokenizer = AutoTokenizer.from_pretrained(model_id)
log_path = args.log_path
save_path = args.save_path

def tokenize(example):
    encoded_inputs = tokenizer(
        example['input'],
        padding='longest',
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoded_inputs['input_ids']
    result = {
        "input_ids": input_ids
    }
    if "output" in example:
        encoded_outputs = tokenizer(
            example['output'],
            padding='longest',
            max_length=512,
            return_tensors='pt'
        )
        label_ids = encoded_outputs['input_ids']
        
        label_ids[label_ids == tokenizer.pad_token_id] = -100
        result.update({
                "labels": label_ids,
            })
        
    return result

if 'xl' in args.model_id:
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True)
    base_model = prepare_model_for_int8_training(base_model)
else:
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
train_datapoints = load_dataset(dataset_name)
train_datapoints = list_of_dicts_to_dict_of_lists(train_datapoints)
train_dataset = Dataset.from_dict(train_datapoints)


tokenized_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))

lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.1,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)


model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(log_path, dataset_name),
    learning_rate=5e-4,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.epoch,
    logging_steps=50,
    # evaluation_strategy='steps',
    # save_total_limit=1,
    # load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # eval_dataset=tokenized_dataset
)
trainer.train()
peft_model_id=os.path.join(save_path, dataset_name)
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

