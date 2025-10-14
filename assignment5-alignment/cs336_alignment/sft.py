from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
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