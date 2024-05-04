import importlib, os, json
from pathlib import Path
from typing import Sequence

from transformers import PreTrainedTokenizer
import datasets
import torch

import numpy as np

def prepare_dataset(dname, hf_path, subset, data_mapper_path, split="validation", debug=False):
    dataset = datasets.load_dataset(hf_path, subset, split=split)
    # Apply the function to each sample using map
    map_fn = getattr(load_module_from_py_file(data_mapper_path), f"add_prefix_{dname}")
    dataset = dataset.map(map_fn, remove_columns=dataset.column_names)
    true_answers = dataset["label"]

    if debug:
        dataset = dataset.select(range(100))
        true_answers = true_answers[:100]
    return dataset, true_answers

def prepare_dataloader(dataset, max_length, b_size, tokenizer, collate_fn):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=b_size, sampler=sampler,
                    collate_fn=lambda x: collate_fn(batch=x, tokenizer=tokenizer, model_max_length=max_length))
    return dataloader

def collate_fn(batch, tokenizer, model_max_length):
    """
    Collate to max length of the batch, if max length is greater than allowed model max length, pad to model max length
    """
    max_len = max([len(tokenizer.encode(x["input_seq"])) for x in batch])
    max_len = min(max_len, model_max_length)
    inputs = tokenizer([b["input_seq"] for b in batch], padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    
    return inputs, [b["label"] for b in batch]


def collate_fn_ddp(batch, tokenizer, model_max_length):
    """
    Collate to max length of the batch, if max length is greater than allowed model max length, pad to model max length
    """
    max_len = max([len(tokenizer.encode(x["input_seq"])) for x in batch])
    max_len = min(max_len, model_max_length)
    inputs = tokenizer([b["input_seq"] for b in batch], padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    # labels = tokenizer([b["label"] for b in batch], padding=True, truncation=True, return_tensors='pt')
    
    return inputs, [b["label"] for b in batch], [b["input_seq"] for b in batch]

def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)
    return module

def calculate_avg_tokens(text_list: Sequence[str], tokenizer: PreTrainedTokenizer, avg: bool) -> int:
    total_tokens = 0
    max_lens = []
    for text in text_list:
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)
        max_lens.append(len(tokens))

    avg_tokens = total_tokens / len(text_list)
    max_tokens = max(max_lens)
    return int(avg_tokens) if avg else max_tokens

def get_max_length(config):
    """
    Extracts maximum token length from the model configuration

    :param model: Hugging Face model
    """

    # Pull model configuration
    # Initialize a "max_length" variable to store maximum sequence length as null
    max_length = None
    # Find maximum sequence length in the model configuration and save it in "max_length" if found
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]: # `d_ff`?
        max_length = getattr(config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    # Set "max_length" to 1024 (default value) if maximum sequence length is not found in the model configuration
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def evaluate_lm(bertscore, rougescore, chrfscore, meteorscore, preds,
                truth, logger, output, results_file_path):

    rouge = rougescore.compute(predictions=preds, references=truth)
    logger.info(rouge)

    chrf = chrfscore.compute(predictions=preds, references=truth)
    logger.info(f"CHRFScore : {chrf}")

    bert = bertscore.compute(predictions=preds, references=truth, lang="en")
    bert.update({
        'precision': np.nanmean(bert['precision']),
        'recall': np.nanmean(bert['recall']),
        'f1': np.nanmean(bert['f1']),
    })
    meteor = meteorscore.compute(predictions=preds, references=truth)
    
    # Dictionary to hold all the scores
    evaluation_results = {
        'bertscore': bert,
        'rouge': rouge,
        'chrf': chrf,
        'meteor': meteor,
    }
    
    if not os.path.exists(output):
        os.makedirs(output)
    
    # Write the evaluation results to a JSON file
    with open(output + results_file_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    logger.info(f"Results saved to {results_file_path}")