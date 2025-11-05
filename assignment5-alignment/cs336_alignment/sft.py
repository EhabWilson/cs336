import os
from tqdm import tqdm
from itertools import islice
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

import wandb

from cs336_alignment.utils import *
from cs336_alignment.math_baseline import *
from cs336_alignment.drgrpo_grader import question_only_reward_fn, extract_answer

# model = AutoModelForCausalLM.from_pretrained(
#     QWEN_MODEL,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
# )
# tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    ids = []
    response_mask = []
    
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_id = tokenizer(prompt, add_special_tokens=False)['input_ids']
        output_id = tokenizer(output, add_special_tokens=False)['input_ids']
        ids.append(torch.LongTensor(prompt_id + output_id))
        response_mask.append(torch.Tensor([False]*len(prompt_id)+[True]*len(output_id)))
        
    ids = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    response_mask = pad_sequence(response_mask, batch_first=True, padding_value=False)

    return {
        "input_ids": ids[:,:-1],
        "labels": ids[:, 1:],
        "response_mask": response_mask[:, 1:]
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    lse = torch.logsumexp(logits, dim=-1)   # (b, l)

    probs = torch.exp(logits - lse[..., None])

    return lse - torch.sum(probs * logits, dim=-1)

def get_response_log_probs(
    model,
    input_ids,
    labels,
    return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:
    output = model(input_ids)
    logits = output.logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    results = {"log_probs": log_probs}

    if return_token_entropy:
        results["token_entropy"] = compute_entropy(logits)

    return results

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    return torch.sum(tensor * mask, dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:    
    loss = masked_normalize(-policy_log_probs, response_mask, normalize_constant)

    loss = loss / policy_log_probs.shape[0] / gradient_accumulation_steps
    loss.backward()

    metadata = {}

    return loss.detach().item(), metadata

def log_generations():
    raise NotImplementedError()

def main(epochs=10, data_slice=128, batch_size=4, gradient_accumulation_steps=4, save_per_epochs=10, lr=5e-5, device="cuda"):
    run = wandb.init(
        project="cs336_cot",
        name=f"{data_slice:04d}",
        # Track hyperparameters and run metadata.
        config={
            "data_slice": data_slice,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "epochs": epochs,
        },
        reinit=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    ds = load_dataset(
        "parquet", 
        data_files={
            "train": "/root/autodl-tmp/data/MATH/data/train-00000-of-00001.parquet",
            "test": "/root/autodl-tmp/data/MATH/data/test-00000-of-00001.parquet",
        },
        split="train",
        # cache_dir="/root/autodl-tmp/hf_cache",
        streaming=True
    )
    dataloader = DataLoader(ds, batch_size)

    for epoch in tqdm(range(epochs), ncols=50):
        total_loss = 0
        for iteration, batch in enumerate(
            islice(dataloader, data_slice//batch_size)
        ):            
            prompts = batch["problem"]
            outputs = batch["solution"]
            data = tokenize_prompt_and_output(prompts, outputs, tokenizer)

            log_probs = get_response_log_probs(
                model=model,
                input_ids=data["input_ids"].to(device),
                labels=data["labels"].to(device)
            )["log_probs"]

            loss, _ = sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=data["response_mask"].to(device),
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            total_loss += loss

            if (iteration + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        run.log({"loss": total_loss})
        if (epoch + 1) % save_per_epochs == 0:
            model.save_pretrained(save_directory=f"/root/autodl-tmp/exps/sft/num_{data_slice:04d}/{epoch + 1:05d}")
            # tokenizer.save_pretrained(save_directory=f"/root/autodl-tmp/exps/sft/{epoch + 1:05d}")

def load_policy_into_vllm_instance(model, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    model.eval()
    model.tie_weights()
    cpu_sd = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}

    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(cpu_sd.items())

def convert_cot_to_think_answer(cot):
    return cot + " </think> <answer> " + extract_answer(cot) + " </answer>"

def load_and_format_prompts(data_path: str) -> tuple[list[str], list[str], list[str]]:
    with open(R1_ZERO_PROMPT, "r") as file:
        prompt = file.read()

    prompts = []
    gts = []
    with open(data_path, "r") as file:
        for line in file:
            data = json.loads(line)
            # prompts.append(data["problem"])
            # gts.append(data["solution"])

            prompts.append(prompt.format(question=data["problem"]))
            gts.append(convert_cot_to_think_answer(data["solution"]))
            # answers.append(extract_answer(data["answer"]))

    return prompts, gts

if __name__ == '__main__':
    # for data_slice in [128, 256, 512, 1024, 5000]:
    # for data_slice in [1024, 5000]:
    #     main(epochs=30, data_slice=data_slice, batch_size=2, gradient_accumulation_steps=8)
    

    prompts, gts = load_and_format_prompts("/root/workspace/cs336/assignment5-alignment/MATH/test.jsonl")
    # llm = LLM(QWEN_MODEL)
    for data_slice in [128, 256, 512, 1024, 5000]:
        epoch=20
        main(epochs=epoch, data_slice=data_slice, batch_size=2, gradient_accumulation_steps=8)
        torch.cuda.empty_cache()
        
        llm = LLM(
            model=f"/root/autodl-tmp/exps/sft_cot/num_{data_slice:04d}/{epoch:05d}",
            tokenizer=QWEN_MODEL,
            gpu_memory_utilization=0.9,
        )
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True
        )
        evaluate_vllm(llm, r1_zero_reward_fn, prompts, gts, sampling_params, save_path=f"/root/autodl-tmp/exps/sft_cot/num_{data_slice:04d}/results.jsonl")
        del llm
        torch.cuda.empty_cache()