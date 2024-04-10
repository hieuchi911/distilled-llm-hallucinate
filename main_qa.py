from tqdm.auto import tqdm
from typing import Sequence, NamedTuple
import logging
import json
import argparse
import os

import numpy as np
import evaluate
import datasets
from transformers import pipeline, AutoTokenizer, AutoConfig
import torch
from transformers.pipelines.pt_utils import KeyDataset

from data_utils import load_module_from_py_file, calculate_avg_tokens

logging.basicConfig(filename='evalscript',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running EvalScript")

logger = logging.getLogger('evalscript')

class LLMTokenizerMapping(NamedTuple):
    """Store data about tokenizers used for an LLM"""
    language_model: str
    tokenizer: str

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bertscore = evaluate.load("bertscore")
    rougescore = evaluate.load('rouge')
    chrfscore = evaluate.load("chrf")
    meteorscore = evaluate.load('meteor')

    dataset = datasets.load_dataset(args.dataset, args.subset, split="validation")
    # Apply the function to each sample using map
    map_fn = getattr(load_module_from_py_file(args.data_mapper), f"add_prefix_{args.dataset}")
    dataset = dataset.map(map_fn)
    true_answers = dataset["label"]

    if args.debug:
        dataset = dataset.select(range(10))
        true_answers = true_answers[:10]

    with open(args.models, "r") as f:
        model_list = json.load(f)
    models: Sequence[LLMTokenizerMapping] = [LLMTokenizerMapping(**model) for model in model_list]
    
    for item in models:
        language_model = item.language_model
        tokenizer = item.tokenizer
        logger.info(language_model)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer, padding_side='left')
        config = AutoConfig.from_pretrained(language_model)
        
        MAX_NEW_TOKENS = calculate_avg_tokens(true_answers, tokenizer, args.avg_new_tokens)
        pipe = pipeline(
            model=language_model,
            tokenizer = tokenizer,
            device=device,
            max_new_tokens=MAX_NEW_TOKENS,
            return_full_text=False if not config.is_encoder_decoder else None
        )
        pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

        summaries = []
        for out in tqdm(pipe(KeyDataset(dataset, "input"), batch_size=args.batch_size), total=len(dataset)):
            summaries.append(out[0]["generated_text"].strip())

        rouge = rougescore.compute(predictions=summaries, references=true_answers)
        logger.info(rouge)

        chrf = chrfscore.compute(predictions=summaries, references=true_answers)
        logger.info(f"CHRFScore : {chrf}")

        bert = bertscore.compute(predictions=summaries, references=true_answers, lang="en")
        bert.update({
            'precision': np.nanmean(bert['precision']),
            'recall': np.nanmean(bert['recall']),
            'f1': np.nanmean(bert['f1']),
        })
        meteor = meteorscore.compute(predictions=summaries, references=true_answers)
        
        # Dictionary to hold all the scores
        evaluation_results = {
            'bertscore': bert,
            'rouge': rouge,
            'chrf': chrf,
            'meteor': meteor,
        }
        
        # Specify the path to the JSON file where you want to save the results
        model_name = language_model.split("/")[1].split("-")[0]
        if "uld_loss" in language_model:
            results_file_path = args.output + f'/evalscript_{args.dataset}_{model_name}_uld_loss.json'
        elif "text_teacher" in language_model:
            results_file_path = args.output + f'/evalscript_{args.dataset}_{model_name}_text_teacher.json'
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        
        # Write the evaluation results to a JSON file
        with open(results_file_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        logger.info(f"Results saved to {results_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract relations (triples) from a file of lecture notes")

    parser.add_argument("-d", "--dataset", type=str, default="hotpot_qa",
                        help="The path to the dataset used to evaluate the student models: hotpot_qa or truthful_qa")
    parser.add_argument("-sub", "--subset", type=str, default="distractor",
                        help="The corresponding subset of the dataset: 'distractor' for hotpot_qa or\
                            'generation' for truthful_qa")
    parser.add_argument("-ms", "--models", type=str, default="./models.json",
                        help="The path to the json file that specifies student model paths")
    parser.add_argument("-dm", "--data_mapper", type=str, default="./data_map.py",
                        help="The path to file that contains the data mappers")
    parser.add_argument("-a", "--avg_new_tokens", action='store_true', 
                        help="Whether to use average length of all answers in the dataset as the number of\
                                new tokens generate - if False then max length of all answers in the dataset")
    parser.add_argument("-db", "--debug", action='store_true', 
                        help="Whether or not to enable debug mode with toy dataset")
    parser.add_argument("-bs", "--batch_size", type=int, default=32,
                        help="Batch size for the data loader")
    parser.add_argument("-o", "--output", type=str, default="./eval_results",
                        help="The path to the folder to store evaluation output")

    args = parser.parse_args()
    
    main(args)
