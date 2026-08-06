import glob
import os
import nltk
import json
import argparse
import random

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from langdetect import detect
from ollama import Client
from pydantic import BaseModel, Field
from typing import List, Dict
from collections import defaultdict
from prompts import (
    SYS_PROMPT_QA_GENERATION,
    USR_PROMPT_QA_GENERATION,
    SYS_PROMPT_TEST_QA_GENERATION,
    USR_PROMPT_TEST_QA_GENERATION
)
from utils import save_jsonl

class QA(BaseModel):
    instruction: str = Field(description="Question extracted from the text")
    output: str = Field(description="Answer extracted from the text")

class QAGeneration(BaseModel):
    qa_pair: List[QA] = Field(description="List of QA Pairs extracted from the text")

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1600, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def table_to_markdown(table):
        rows = table.find_all("tr")
        if not rows:
            return ""
        
        md_rows = []
        for i, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            cell_texts = [c.get_text(strip=True) for c in cells]
            md_rows.append("| " + " | ".join(cell_texts) + " |")
            
            if i == 0:
                md_rows.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")
        
        return "\n".join(md_rows)
    
    def clean_html(self, content: str):
        soup = BeautifulSoup(content, "html.parser")
        title = soup.title.string if soup.title else None

        if soup.head:
            soup.head.decompose()

        for tag in soup(["aside", "style", "footer", "script", "img", "iframe", "form", "noscript", "a", "button"]):
            tag.decompose()

        selectors = [
            "div.skipnav",
            "div.toolbar.top-nav-on",
            "div.section-toc.section-toc-before",
            "div.breadcrumb-container",
            "div.top-pager",
            "div.feedback-panel"
        ]
        
        for selector in selectors:
            for element in soup.select(selector):
                element.decompose()

        for table in soup.find_all("table"):
            md = self.table_to_markdown(table)
            table.replace_with(md)

        return title, soup.get_text(separator="\n", strip=True)

    def lemmatize(self, text: str):
        if not text:
            return ""
        tokens = word_tokenize(text.lower())
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        return " ".join(lemmatized)

    def process_files(self, paths: List[str], split: bool = False):
        data = []

        for path in tqdm(paths, desc="Processing Files"):
            path_obj = Path(path)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if path_obj.suffix == ".html":
                title, text = self.clean_html(content)
            else:
                title, text = None, content.strip()
            
            lang = path_obj.parent.name
            if len(lang) != 2 or not lang.isalpha():
                try:
                    lang = detect(text)
                except Exception as e:
                    lang = "en"
            
            chunks = self.splitter.split_text(text) if split else [text]

            for chunk in chunks:
                data.append({
                    "text": chunk,
                    "lemmatized_text": self.lemmatize(chunk),
                    "metadata": {
                        "title": title or path_obj.stem,
                        "language": lang,
                        "file_path": str(path)
                    }
                })
        
        return data
            

class DatasetGenerator:
    def __init__(self, model_name: str, host: str):
        self.client = Client(host=host)
        self.model = model_name

    def _unload_model(self):
        self.client.chat(model=self.model, messages=[{"role": "user", "content": ""}], keep_alive=0)
    
    def generate_qa_batch(
        self,
        dataset: List[Dict],
        output_path: str,
        sys_prompt: str,
        usr_prompt: str
    ):
        results = []
        path = Path(output_path)

        with open(path, "a", encoding="utf-8") as f:
            for i, item in enumerate(tqdm(dataset, desc="Extracting QA Pairs")):
                if (i + 1) % 500 == 0:
                    self._unload_model()
                
                prompt = usr_prompt.format(
                    title=item["metadata"]["title"],
                    text=item["text"]
                )

                try:
                    response = self.client.chat(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        format=QAGeneration.model_json_schema(),
                        options={
                            "num_ctx": 8192,
                            "temperature": 0.4
                        }
                    )

                    data = QAGeneration(**json.loads(response["message"]["content"]))

                    for qa in data.qa_pair:
                        result = {
                            "instruction": qa.instruction,
                            "output": qa.output,
                            "metadata": item["metadata"]
                        }
                        results.append(result)
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                except Exception as e:
                    print(f"Error creating QA Pairs for {item['metadata']['file_path']}: {e}")

        return results
    
    def generate_test_set(
        self,
        sft_dataset: List[Dict],
        text_dataset: List[Dict],
        test_size: float,
        output_path: str
    ):
        training_map = defaultdict(list)
        for item in sft_dataset:
            training_map[item["metadata"]["file_path"]].append(item)
        
        raw_text_map = defaultdict(list)
        for item in text_dataset:
            raw_text_map[item["metadata"]["file_path"]].append(item)
        
        paths = list(training_map.keys())
        alarms_paths = [p for p in paths if "alarms" in p.lower()]
        manual_paths = [p for p in paths if "alarms" not in p.lower()]

        test_set_size = int(len(paths) * test_size)
        half_test = test_set_size // 2

        selected_test_paths = (
            random.sample(alarms_paths, min(len(alarms_paths), half_test)) +
            random.sample(manual_paths, min(len(manual_paths), half_test))
        )

        test_results = []
        with open(output_path, "a", encoding="utf-8") as f:
            for i, path in enumerate(tqdm(selected_test_paths, desc="Generating Test Set")):
                if (i + 1) % 500 == 0:
                    self._unload_model()
                
                excluded_questions = [qa["instruction"] for qa in training_map[path]]
                full_text = "\n".join([item["text"] for item in raw_text_map[path]])
                title = training_map[path][0]["metadata"].get("title", "Document")

                prompt = USR_PROMPT_TEST_QA_GENERATION.format(
                    title=title,
                    original_text=full_text,
                    excluded_questions="\n- ".join(excluded_questions)
                )

                try:
                    response = self.client.chat(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYS_PROMPT_TEST_QA_GENERATION},
                            {"role": "user", "content": prompt}
                        ],
                        format=QAGeneration.model_json_schema(),
                        options={
                            "num_ctx": 8192,
                            "temperature": 0.4
                        }
                    )

                    data = QAGeneration(**json.loads(response["message"]["content"]))
                    for qa in data.qa_pair:
                        result = {
                            "instruction": qa.instruction, 
                            "output": qa.output, 
                            "metadata": {
                                "file_path": path, 
                                "title": title
                            }
                        }
                        test_results.append(result)
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                except Exception as e:
                    print(f"Error creating QA Pairs for {path}: {e}")
        
        return test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root folder Alarms and Manuals")
    parser.add_argument("--model_ollama", type=str, required=True, help="Ollama Model for creating QA pairs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder JSONL files.")
    parser.add_argument("--force", action="store_true", help="Force rewriting JSONL files.")
    parser.add_argument("--host_ollama", type=int, default=11434, help="Ollama Host")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test Size")

    args = parser.parse_args()

    nltk.download('punkt_tab')
    nltk.download('wordnet')

    root = Path(args.root)
    alarms = root / "Alarms"
    manuals = root / "Manuals"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alarms_jsonl = output_dir / "alarms_dataset.jsonl"
    manuals_jsonl = output_dir / "manuals_dataset.jsonl"
    full_jsonl = output_dir / "full_dataset.jsonl"
    sft_jsonl = output_dir / "sft_dataset_full.jsonl"
    test_jsonl = output_dir / "test_set.jsonl"

    if args.force:
        files_to_remove = [
            alarms_jsonl,
            manuals_jsonl,
            full_jsonl,
            sft_jsonl,
            test_jsonl
        ]

        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
        
        print("Force is True. Previous JSONL files removed.")
    
    valid_ext = ["html", "txt"]
    alarms_paths = []
    manuals_paths = []

    for file_path in glob.iglob(str(alarms / "**" / "*"), recursive=True):
        if os.path.isfile(file_path) and file_path.split(".")[-1] in valid_ext:
            alarms_paths.append(file_path)

    for file_path in glob.iglob(str(manuals / "**" / "*"), recursive=True):
        if os.path.isfile(file_path) and file_path.split(".")[-1] in valid_ext:
            manuals_paths.append(file_path)

    processor = DocumentProcessor()
    alarms_data = processor.process_files([str(path) for path in alarms_paths], split=True)
    manuals_data = processor.process_files([str(path) for path in manuals_paths], split=True)    
    full_data = alarms_data + manuals_data

    save_jsonl(alarms_data, alarms_jsonl)
    save_jsonl(manuals_data, manuals_jsonl)
    save_jsonl(full_data, full_jsonl)

    generator = DatasetGenerator(
        model_name=args.model_ollama,
        host=f"http://localhost:{args.host_ollama}"
    )

    sft_data = generator.generate_qa_batch(
        dataset=full_data,
        output_path=str(sft_jsonl),
        sys_prompt=SYS_PROMPT_QA_GENERATION,
        usr_prompt=USR_PROMPT_QA_GENERATION
    )

    test_set = generator.generate_test_set(
        sft_dataset=sft_data,
        text_dataset=full_data,
        test_size=args.test_size,
        output_path=str(test_jsonl)
    )