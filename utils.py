import json

from typing import List, Dict
from datasets import Dataset

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def get_dataset(dataset_path: str):
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return Dataset.from_list(data)