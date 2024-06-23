from lorahub.algorithm import lorahub_learning, lorahub_inference
from lorahub.constant import LORA_MODULE_NAMES, LORA_MODULE_NAMES_XL
import random
import argparse
from inference_utils import get_examples_for_inference, get_examples_for_learning
import os
import json


def get_lora_module_list(args):
    if args.lora == 'large':                 
        return LORA_MODULE_NAMES
    
    elif args.lora == 'xl':          
        return LORA_MODULE_NAMES_XL


def main(args):
    """
    Perform lorahub learning
    """
    # get a list of modules to be used in the composition

    modules = get_lora_module_list(args)
    print("modules:", modules)
    
    # construct input list and output list
    example_inputs, examples_outputs = [], []
    for example in get_examples_for_learning(args.dataset, args.seed, args.num):
        example_inputs.append(example["input"])
        examples_outputs.append(example["output"])

    # perform LoRAHub learning
    module_weights, model, tokenizer = lorahub_learning(lora_module_list=modules,
                                                        example_inputs=example_inputs,
                                                        example_outputs=examples_outputs,
                                                        max_inference_step=40,
                                                        batch_size=5)

    print("module_weights:", module_weights)
 
    """
    Perform inference to get predictions
    """
    # now you can use the model to perform inference
    example_inputs, examples_outputs = [], []
    for example in get_examples_for_inference(args.dataset):
        example_inputs.append(example["input"])
        examples_outputs.append(example["output"])

    example_predictions = lorahub_inference(example_inputs=example_inputs,
                                            model_or_name_path=model,
                                            tokenizer_or_tokenizer_path=tokenizer,
                                            batch_size=args.inference_batch_size,
                                            # can set as None if you do not have the ground truth
                                            example_outputs=examples_outputs)
    results = [{'prediction': p, 'label': l} for p, l in zip(example_predictions, examples_outputs)]
    

    with open(os.path.join(args.output_path, '{}.json'.format(args.dataset)), 'w') as f:
        json.dump(results, f ,indent=2)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, default='commonsense_qa')
    arg_parser.add_argument('--lora', type=str, default='xl')
    arg_parser.add_argument('--output_path', type=str, default='results')
    arg_parser.add_argument('--inference_batch_size', type=int, default=16)
    arg_parser.add_argument('--lora_sample', type=int, default=0)


    args = arg_parser.parse_args()
    main(args)

