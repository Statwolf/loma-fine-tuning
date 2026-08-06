# Loma Fine Tuning

Fine-tuning pipeline for **Loma**, a technical assistant fine-tuned on QA pairs extracted from technical manuals and alarm documentation. Supports LoRA training with custom keyword tokens, GGUF export, and local serving via Ollama.

## Prerequisites

- [Ollama](https://ollama.com/) running locally
- [Qdrant](https://qdrant.tech/) for the knowledge base (optional)
- Docker (for GGUF quantization)
- Python 3.10

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
make preprocess   # Process raw docs into JSONL datasets
make extract-kw   # Extract technical keywords
make train        # Fine-tune the model
```

## Workflow

```
raw docs (HTML/TXT)
  ├──→ preprocess       → JSONL datasets (alarms, manuals, full, sft, test)
  └──→ extract-kw       → keywords.jsonl (custom tokens for the vocab)

keywords + sft dataset
  └──→ train            → LoRA adapters → merged model

merged model
  ├──→ gguf             → GGUF format
  └──→ quantize-gguf    → quantized GGUF
```

## Usage

### Preprocess

Scans the root directory for `Alarms/` and `Manuals/` subfolders, cleans HTML, chunks text, detects language, and generates QA pairs using an Ollama model.

```bash
make preprocess
```

| Argument | Description | Default |
|---|---|---|
| `--root` | Root folder with `Alarms/` and `Manuals/` | (set in Makefile) |
| `--model_ollama` | Ollama model for QA generation | `gemma4:26b` |
| `--host_ollama` | Ollama server port | `6000` |
| `--test_size` | Fraction of documents for test set | `0.1` |
| `--force` | Delete existing JSONL files before processing | yes |

**Output files** (in `./data/`):

| File | Description |
|---|---|
| `alarms_dataset.jsonl` | Chunked alarm documents |
| `manuals_dataset.jsonl` | Chunked manual documents |
| `full_dataset.jsonl` | Combined alarm + manual documents |
| `sft_dataset_full.jsonl` | Generated instruction-following pairs |
| `test_set.jsonl` | Held-out test QA pairs (balanced alarms/manuals) |

### Keyword Extraction

Extracts technical keywords using KeyBERT with word-frequency penalty, regex-matched document codes, and MinMax normalization.

```bash
make extract-kw
```

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Input dataset path | `./data/test.jsonl` |
| `--kw_path` | Output keywords file | `./data/keywords.jsonl` |
| `--top_k` | Max keywords to extract | `600` |
| `--thr` | Score threshold | `0.3` |
| `--model_ollama` | Ollama model for keyword correction | `gemma4:26b` |

### Fine-Tuning

```bash
make train
```

| Argument | Description | Default |
|---|---|---|
| `--model` | Base model | `unsloth/Qwen3.5-9B` |
| `--dataset` | Training dataset path | `./data/test.jsonl` |
| `--test_dataset` | Test dataset path | `./data/test_set.jsonl` |
| `--output_dir` | Output directory | `./results/{ID}_r{r}_lr{lr}` |
| `--unsloth` | Use Unsloth optimization | yes |
| `--quantize` | Load model in 4-bit (QLoRA) | yes |
| `--add_tokens` | Add keywords to vocabulary | yes |
| `--instruct` | Instruction-finetuning mode | yes |
| `--tokens_path` | Keywords JSONL file | `./data/keywords.jsonl` |
| `--lora_r` | LoRA rank | `128` |
| `--lora_alpha` | LoRA alpha | `256` |
| `--lr` | Learning rate | `2e-4` |
| `--batch_size` | Per-device batch size | `2` |
| `--grad_acc` | Gradient accumulation steps | `4` |
| `--clipping` | Gradient clipping norm | `0.4` |
| `--max_length` | Max sequence length | `4096` |
| `--epochs` | Number of training epochs | `4` |
| `--device` | CUDA device | `2` |
| `--id` | Run identifier (used in output name) | `test` |

**Training tracking**: W&B at `wandb`, (project: `loma-fine-tuning`, entity: `loma-statwolf`).

### Merge & Export

```bash
make merge           # Merge LoRA adapter into base model
make gguf            # Convert to GGUF format
make quantize-gguf   # Quantize GGUF (runs via Docker)
```

### Custom Run

Override any Makefile variable on the command line:

```bash
make train ID=run01 EPOCHS=5 LR=1e-4 LORA_R=256
```

## Architecture

| Module | Class | Purpose |
|---|---|---|
| `preprocess.py` | `DocumentProcessor` | HTML cleaning, chunking, lemmatization |
| `preprocess.py` | `DatasetGenerator` | QA pair generation via Ollama |
| `keywords_extraction.py` | `KeywordExtractor` | KeyBERT scoring + LLM correction |
| `train.py` | `TokenHandler` | Vocabulary extension + embedding init |
| `train.py` | `ModelLoader` | Model loading (Unsloth / HF) |
| `train.py` | `FineTuning` | Training orchestration |
| `kb.py` | `KnowledgeBaseLoma` | Qdrant knowledge base management |

## Output Layout

```
results/
  └── test_r128_lr2e-4/          # LoRA adapter output
  └── test_r128_lr2e-4_adapters/  # Adapter weights + tokenizer
  └── test_r128_lr2e-4_merged/   # Merged full model
  └── test_r128_lr2e-4_q8_0.gguf   # GGUF model
data/
  └── keywords.jsonl             # Custom vocabulary tokens
```
