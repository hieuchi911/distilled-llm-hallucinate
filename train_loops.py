import torch
import torch.distributed as dist

import os
import debugpy
import tqdm

def train_loop(model, tokenizer, dataloader, max_new_tokens, prompt_kw, device, is_encoder_decoder):
    model.to(device)
    model.eval()

    predictions, ground_truths = [], []
    
    with torch.no_grad():
        for batch, truths in tqdm(dataloader, total=len(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens, use_cache=True,
                            pad_token_id=tokenizer.pad_token_id)
            generated_text_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # ignore text longer than model context length
            for generated_text, truth, inp_id in zip(generated_text_batch, truths, input_ids):
                if prompt_kw not in tokenizer.decode(inp_id):
                    count += 1
                    continue
                generated_preds = generated_text if is_encoder_decoder else generated_text.split(prompt_kw)[1]
                predictions.append(generated_preds)
                ground_truths.append(truth)
                
            # clean cache
            torch.cuda.empty_cache()
    return predictions, ground_truths, count

class GatherObject:
    def __init__(self, preds: torch.tensor, truth: torch.tensor, inp: torch.tensor) -> None:
        self.preds_ids = preds
        self.truth_text = truth
        self.inp_text = inp


def train_loop_ddp(model, tokenizer, dataloader, max_new_tokens,
                    prompt_kw, device, is_encoder_decoder):

    rank = int(os.environ["RANK"]) # dist.get_rank()
    world_size = int(os.environ["WORLD_SIZE"]) # dist.get_world_size()

    local_rank = int(os.environ["LOCAL_RANK"]) # dist.get_rank()
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"]) # dist.get_world_size()
    # import time
    # time.sleep(10)  # adjust the sleep time as needed

    # port = 5678 + rank

    # # Start the debug server
    # debugpy.listen(('0.0.0.0', port))

    # print(f"Process {dist.get_rank()} waiting for debugger to attach on port {port}...")
    # debugpy.wait_for_client()

    model.to(device)
    model.eval()

    predictions, ground_truths, count = [], [], 0
    
    with torch.no_grad():
        for i, (batch, truth_text, inp_text) in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # truths_ = truths["input_ids"].to(device)
            
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, use_cache=True,
                            pad_token_id=tokenizer.pad_token_id)
            gather_obj = GatherObject(outputs, truth_text, inp_text)

            # gather outputs from all GPUs and store in gathered_outputs using dist.gather()
            gathered_outputs = [None for _ in range(world_size)] if rank == 0 else None
            # gather all outputs from all GPUs and store in gathered_outputs in device 0
            print(f"GPU - {rank}:({local_rank}) next batch (index {i})- {input_ids.shape} - {outputs.shape}")
            dist.gather_object(gather_obj, gathered_outputs, dst=0)
            
            if rank == 0:
                generated_text_batch = []
                truth_text_batch = []
                inp_text_batch = []
                for i in range(world_size):
                    generated_text_batch.extend(tokenizer.batch_decode(gathered_outputs[i].preds_ids, skip_special_tokens=True))
                    truth_text_batch.extend(gathered_outputs[i].truth_text)
                    inp_text_batch.extend(gathered_outputs[i].inp_text)
                # ignore text longer than model context length
                for generated_text, truth, inp_text in zip(generated_text_batch, truth_text_batch, inp_text_batch):
                    if prompt_kw not in inp_text:
                        count += 1
                        continue
                    generated_preds = generated_text if is_encoder_decoder else generated_text.split(prompt_kw)[1]
                    predictions.append(generated_preds)
                    ground_truths.append(truth)
            # clean cache
            torch.cuda.empty_cache()
    return predictions, ground_truths, count