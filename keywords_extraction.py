import json
import numpy as np
import argparse
import re

from tqdm import tqdm
from wordfreq import word_frequency
from keybert import KeyBERT
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from pydantic import BaseModel, Field
from ollama import Client
from utils import load_jsonl, save_jsonl
from typing import List, Dict

EPSILON = 1e-15

class FixKeyword(BaseModel):
    keyword: str = Field(
        description="Keyword fixed if broken, otherwise the same identical keyword."
    )

class KeywordExtractor:
    def __init__(
        self,
        ollama_model: str,
        ollama_host: str = "http://localhost:11434", 
        st_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.st_model = SentenceTransformer(st_model_name)
        self.kw_model = KeyBERT(model=self.st_model)
        self.ollama_client = Client(host=ollama_host)
        self.ollama_model = ollama_model
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _get_raw_keybert_scores(self, data: List[Dict]):
        tmp_data = defaultdict(lambda: {"scores": [], "language": ""})

        for item in tqdm(data, desc="Extracting Keywords"):
            text = item.get("text", "")
            if not text:
                continue

            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 1),
                stop_words=None,
                use_mmr=True
            )

            lang = item.get("metadata", {}).get("language", "en")
            for key, score in keywords:
                tmp_data[key]["scores"].append(score)
                tmp_data[key]["language"] = lang
        
        return [
            {
                "keyword": k,
                "avg_score": float(np.mean(v["scores"])),
                "language": v["language"],
                "occurrence": len(v["scores"])
            }
            for k, v in tmp_data.items()
        ]

    def _apply_word_frequency_penalty(self, keywords: List[Dict], codes_set: set):
        scored_results = []

        for item in keywords:
            if item["keyword"] in codes_set:
                continue

            freq = word_frequency(item["keyword"], item["language"])
            final_score = item["avg_score"] / (freq + EPSILON)
            scored_results.append((item["keyword"], final_score))
        
        return scored_results

    @staticmethod
    def _build_ollama_prompt(keyword: str):
        return f"""
            Target: Fix grammatical or syntactic errors in the keyword. 
            Rules:
            1. Separate merged words that should be distinct (e.g., 'manualrestart' -> 'manual restart').
            2. DO NOT divide compound words (e.g., 'Sovrariscaldamento', 'reinigungszyklus').
            3. DO NOT touch or modify technical codes (e.g., 'AA111', 'X-100'). Leave them exactly as they are.
            4. DO NOT separate prefixes or suffixes.
            5. DO NOT split words in German, Dutch, or Nordic languages (e.g., 'reinigungszyklus' is CORRECT, do not change to 'reinigungs zyklus').
            6. Maintain the original language of the keyword.

            Examples:
            - Keyword: Sovrariscaldamento -> Sovrariscaldamento
            - Keyword: AA111 -> AA111
            - Keyword: manualrestart -> manual restart
            - Keyword: reinigungszyklus -> reinigungszyklus

            **Keyword**: {keyword}
            
            Return ONLY the fixed keyword, no explanations.
        """

    def extract_codes(self, data: List[Dict]):
        re_codes = re.compile(r"^[A-Z]{1,3}[0-9]{3,}$")

        codes = set()
        for item in tqdm(data, desc="Extracting codes"):
            title = item.get("metadata", {}).get("title")
            if title and re_codes.match(title):
                codes.add(title)

        return codes

    def fix_with_llm(self, keywords: List[Dict]):
        fixed_results = []

        for item in tqdm(keywords, desc="Fixing Keywords with LLM"):
            prompt = self._build_ollama_prompt(item["keyword"])
            try:
                response = self.ollama_client.chat(
                    model=self.ollama_model,
                    messages=[
                        {"role": "system", "content": "You are a technical editor. You must respond in valid JSON matching the provided schema."},
                        {"role": "user", "content": prompt}
                    ],
                    format=FixKeyword.model_json_schema()
                )
                response_json = json.loads(response["message"]["content"])
                fixed_kw = FixKeyword(**response_json)
                fixed_results.append({
                    "keyword": fixed_kw.keyword,
                    "score": item["score"]
                })
            except Exception as e:
                print(f"Error fixing {item['keyword']}: {e}")
                fixed_results.append(item)
        
        return fixed_results

    def process(
        self, 
        data: List[Dict], 
        top_k: int, 
        threshold: float = 0.7, 
        use_llm_correction: bool = False,
        include_codes: bool = False
    ):
        codes_set = self.extract_codes(data) if include_codes else set()
        raw_keywords = self._get_raw_keybert_scores(data)
        scored_list = self._apply_word_frequency_penalty(raw_keywords, codes_set)

        if not scored_list:
            return []
        
        raw_scores = np.array([x[1] for x in scored_list]).reshape(-1, 1)
        norm_scores = self.scaler.fit_transform(raw_scores)

        final_list = [
            {"keyword": scored_list[i][0], "score": round(float(norm_scores[i][0]), 3)}
            for i in range(len(scored_list))
        ]

        filtered = [kw for kw in final_list if kw["score"] > threshold]
        filtered.sort(key=lambda x: x["score"], reverse=True)
        top_results = filtered[:top_k]

        if use_llm_correction:
            return self.fix_with_llm(top_results)
        
        return top_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="Dataset Path (JSONL file)")
    parser.add_argument("--kw_path", type=str, required=True, help="Keywords final path")
    parser.add_argument("--top_k", type=int, required=True, help="Max number of keywords to extract")
    parser.add_argument("--model_ollama", type=str, required=True, help="Ollama Model for Fixing Keywords")
    parser.add_argument("--thr", type=float, default=0.7, help="Threshold")
    parser.add_argument("--host_ollama", type=int, default=11434, help="Ollama Host")
    args = parser.parse_args()

    data = load_jsonl(args.dataset)

    extractor = KeywordExtractor(
        ollama_model=args.model_ollama,
        ollama_host=f"http://localhost:{args.host_ollama}"
    )
    
    results = extractor.process(
        data=data,
        top_k=args.top_k,
        threshold=args.thr,
        use_llm_correction=False,
        include_codes=False
    )

    save_jsonl(results, args.kw_path)