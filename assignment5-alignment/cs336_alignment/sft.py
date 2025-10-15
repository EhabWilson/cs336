from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .utils import *

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
        ids.append(torch.Tensor(prompt_id + output_id))
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
    pass