import pandas as pd
import json
from cs336_alignment.drgrpo_grader import extract_answer
from cs336_alignment.utils import *
import re

def load_raw_data(file_path: str):
    """从 JSONL 文件加载原始数据。"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line: {line.strip()} - Error: {e}")
    return data

def transform_to_sft_format(raw_data):
    """
    将原始数据转换为 SFT 训练所需的对话格式。
    并拆分 <think> 和 <answer> 部分。
    """
    formatted_records = []
    
    with open(R1_ZERO_PROMPT, "r") as file:
        prompt = file.read()

    for item in raw_data:
        problem = item.get("problem", "")
        solution = item.get("solution", "")
        
        if not problem or not solution:
            continue

        final_answer = extract_answer(solution)
        if not final_answer:
             final_answer = "MISSING_ANSWER" 

        # # 移除 solution 文本中的 \\boxed{...} 结构，作为 <think> 的内容
        # try:
        #     think_process = re.findall(r'\\boxed\{([^}]*)\}', solution)
        # except:
        #     breakpoint()
        # # think_process = re.sub(r'\\boxed\{([^}]*)\}', final_answer, solution).strip()

        formatted_records.append({
            "problem": prompt.format(question=problem),
            "answer": f"<think> {solution} </think> <answer> {final_answer} </answer>"
        })
        
    return formatted_records

def save_to_parquet(data, output_file: str):
    """将数据列表转换为 Pandas DataFrame 并存储为 Parquet 文件。"""
    
    # 转换为 Pandas DataFrame
    df = pd.DataFrame(data)
    
    # 使用 to_parquet 存储
    # engine='pyarrow' 是推荐的 Parquet 引擎
    # compression='snappy' 是 Parquet 文件常用的快速、高效的压缩方式
    df.to_parquet(
        output_file, 
        engine='pyarrow', 
        compression='snappy',
        index=False # 通常不需要存储 Pandas 的索引
    )
    print(f"\n✅ 数据已成功转换并存储到 Parquet 文件: {output_file}")
    print(f"文件包含 {len(df)} 条记录，列名: {df.columns.tolist()}")

if __name__ == '__main__':
    raw_records = load_raw_data("/root/workspace/cs336/assignment5-alignment/MATH/train.jsonl")
    processed_data = transform_to_sft_format(raw_records)
    
    # 4. 存储为 Parquet
    save_to_parquet(processed_data, "/root/workspace/cs336/assignment5-alignment/MATH/train.parquet")
    
    # 5. 验证文件 (可选)
    # print("\n--- Parquet 文件内容验证 ---")
    # df_check = pd.read_parquet("/root/workspace/cs336/assignment5-alignment/MATH/train.parquet")
    # print(df_check[['problem', 'sft_text']].head(1).to_markdown(index=False))