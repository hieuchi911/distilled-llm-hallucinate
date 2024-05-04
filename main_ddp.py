import logging
import argparse
import os

import evaluate
import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from data_utils import calculate_avg_tokens, collate_fn, collate_fn_ddp, prepare_dataset, prepare_dataloader,\
            get_max_length, evaluate_lm, load_module_from_py_file

from model_utils import prepare_ddp_model, prepare_model

from train_loops import train_loop_ddp

logging.basicConfig(filename='evalscript',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running EvalScript")

logger = logging.getLogger('evalscript')

# def main(rank, world_size, args):
def main(args):
    # Initialize the process group
    os.environ['NCCL_DEBUG'] = 'INFO'
    init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    
    # Get the local rank of the process
    device = torch.device("cuda:{}".format(rank))

    bertscore = evaluate.load("bertscore")
    rougescore = evaluate.load('rouge')
    chrfscore = evaluate.load("chrf")
    meteorscore = evaluate.load('meteor')
    
    logger.info(args.language_model)
    print(f"\nStudent model name: {args.language_model}\nEvaluated on {args.dname} dataset\n")
    
    model, tokenizer, config = prepare_ddp_model(args.language_model, args.tokenizer)

    dataset, true_answers = prepare_dataset(args.dname, args.dpath, args.dsubset, args.data_mapper, args.dsplit, args.debug)

    # ensuring input length don't exceed `model_max_length-max_new_tokens`,
    # since `max_new_tokens` will be generated autoregressively
    max_new_tokens = calculate_avg_tokens(true_answers, tokenizer, args.avg_new_tokens)
    model_max_length = get_max_length(config)   # get model context length
    max_length = model_max_length - max_new_tokens

    dataloader = prepare_dataloader(dataset, max_length, args.batch_size, tokenizer, collate_fn_ddp)

    prompt_kws = getattr(load_module_from_py_file(args.data_mapper), "PROMPT_KEYWORDS")
    predictions, ground_truths, count = train_loop_ddp(model=model, tokenizer=tokenizer, dataloader=dataloader,
                                            max_new_tokens=max_new_tokens, prompt_kw=prompt_kws[args.task],
                                            device=device, is_encoder_decoder=config.is_encoder_decoder,
                                            )
    
    if rank == 0:
        if not predictions:   # if none of the inputs is shorter than model's context length
            logger.info(f"there are {count} empty predictions due to ignored long inputs")
            return
            
        # Specify the path to the JSON file where you want to save the results
        model_name = args.language_model.split("/")[1].split("-")[0]
        if "uld_loss" in args.language_model:
            results_file_name = f'evalscript_{args.dname}_{model_name}_uld_loss.json'
        elif "text_teacher" in args.language_model:
            results_file_name = f'evalscript_{args.dname}_{model_name}_text_teacher.json'

        evaluate_lm(bertscore, rougescore, chrfscore, meteorscore, preds=predictions,
                    truth=ground_truths, logger=logger, output=args.output + f"/{args.task}/",
                    results_file_path=results_file_name)
    
    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract relations (triples) from a file of lecture notes")
    
    parser.add_argument("-t", "--task", type=str, default="qa",
                        help="The type of task that the student models are finetuned in: summ, qa")
    parser.add_argument("-dn", "--dname", type=str, default="hotpot_qa",
                        help="The dataset name used to evaluate the student models: hotpot_qa\
                            , truthful_qa, cnn_dm, or qmsum")
    parser.add_argument("-dp", "--dpath", type=str, default="hotpot_qa",
                        help="The huggingface path to the dataset")
    parser.add_argument("-ds", "--dsubset", type=str, default="distractor",
                        help="The corresponding subset of the dataset: e.g. 'distractor' for hotpot_qa,\
                            'generation' for truthful_qa, '1.0.0' for cnn_dm, or 'default' for qmsum")
    parser.add_argument("-dspl", "--dsplit", type=str, default="distractor",
                        help="The data split to use from the dataset")
    parser.add_argument("-max", "--max_len", type=int, default=2000,
                        help="The maximum length of tokenized context") # for hotpot_qa majority of length values is < 1000
    parser.add_argument("-ms", "--models", type=str, default="./models_summ.json",
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
    parser.add_argument("-tok", "--tokenizer", type=str, default="t5-base",
                        help="The name of the tokenizer to use")
    parser.add_argument("-lm", "--language_model", type=str, default="t5-base",
                        help="The name of the language model to use")

    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    # mp.spawn(main, nprocs=world_size, args=(world_size, args))
    main(args)