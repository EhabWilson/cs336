from vllm import LLM, SamplingParams
from typing import Callable, List
import pandas as pd
import json

from drgrpo_grader import r1_zero_reward_fn
from utils import *
from drgrpo_grader import extract_answer


def evaluate_vllm(
    vllm_model: LLM,\
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    gts,
    eval_sampling_params: SamplingParams,
    save_path: str = "math_results.jsonl"
 ) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    format_rewards, answer_rewards, rewards = [], [], []
    with open(save_path, "w", encoding="utf-8") as f:
        for output, gt in zip(outputs, gts):
            generated_answer = output.outputs[0].text
            r = reward_fn(generated_answer, gt)
            format_rewards.append(r["format_reward"])
            answer_rewards.append(r["answer_reward"])
            rewards.append(r["reward"])

            result = {
                "prompt": output.prompt,
                "generated_answer": generated_answer,
                "ground_truth": gt,
                "format_reward": r["format_reward"],
                "answer_reward": r["answer_reward"],
                "reward": r["reward"]
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Average format reward: {sum(format_rewards) / len(format_rewards)}")
        print(f"Average answer reward: {sum(answer_rewards) / len(answer_rewards)}")
        print(f"Average total reward: {sum(rewards) / len(rewards)}")

        result = {
            "Average format reward": sum(format_rewards) / len(format_rewards),
            "Average answer reward": sum(answer_rewards) / len(answer_rewards),
            "Average total reward": sum(rewards) / len(rewards)
        }
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def replace_last_occurrence(text: str, old: str, new: str) -> str:
    """
    替换 text 中最后一次出现的 old 为 new
    """
    idx = text.rfind(old)
    if idx == -1:
        return text  # 如果没找到 old，就返回原字符串
    return text[:idx] + new + text[idx + len(old):]

def parse_MATH_data(data_path: str):
    with open(R1_ZERO_PROMPT, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompts = []
    gts = []
    df = pd.read_parquet(data_path)
    for problem, solution in zip(df.problem, df.solution):
        prompt = replace_last_occurrence(prompt_template, "question", problem)
        prompts.append(prompt)
        gts.append(extract_answer(solution))
    return prompts, gts

if __name__ == '__main__':
    llm = LLM(QWEN_MODEL)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    prompts, gts = parse_MATH_data(MATH_TEST)

    evaluate_vllm(llm, r1_zero_reward_fn, prompts, gts, sampling_params)