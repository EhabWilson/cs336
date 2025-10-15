from datasets import load_dataset

# 加载 GSM8K 数据集（默认是 parquet 格式）
ds = load_dataset("parquet", data_files={
    "train": "/root/autodl-tmp/data/MATH/data/train-00000-of-00001.parquet",
    "test": "/root/autodl-tmp/data/MATH/data/test-00000-of-00001.parquet",
}, cache_dir="/root/autodl-tmp/hf_cache")

# 保存为 JSONL
ds["train"].to_json("MATH/train.jsonl", lines=True)
ds["test"].to_json("MATH/test.jsonl", lines=True)
