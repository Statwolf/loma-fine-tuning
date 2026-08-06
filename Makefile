.PHONY: help preprocess extract-kw train merge gguf quantize-gguf
 
# --- Paths ---
ROOT         = <path_root>
DATASET      = <path_dataset>
TEST_DATASET = <path_test_dataset>
OUTPUT_DIR   = <output_directory>
TOKENS_PATH  = <path_tokens>

# --- Model ---
MODEL        = unsloth/Qwen3.5-9B
MODEL_OLLAMA ?= gemma4:26b
MODEL_QA     ?= gemma4:26b

# --- Preprocessing ---
HOST_OLLAMA ?= 6000
TOP_K       ?= 600
THRESHOLD   ?= 0.3
TEST_SIZE   ?= 0.1
FORCE       ?= 1

# --- Training ---
DEVICE       ?= 2
ID           ?= test
LORA_R       ?= 128
LORA_ALPHA   ?= 256
LR           ?= 2e-4
BATCH_SIZE   ?= 2
GRAD_ACC     ?= 4
GRAD_CLIPPING ?= 0.4
MAX_LENGTH   ?= 4096
EPOCHS       ?= 4
INSTRUCT     ?= 1
TOKENS       ?= 1
UNSLOTH      ?= 1
QUANTIZE     ?= 1
OUTPUT_MODEL_NAME = $(ID)_r$(LORA_R)_lr$(LR)
OUTPUT_MODEL = ./results/$(OUTPUT_MODEL_NAME)

# --- Quantization ---
QUANT_TYPE = bf16

# --- Flags ---
UNSLOTH_FLAG  = $(if $(filter 1,$(UNSLOTH)),--unsloth,)
TOKENS_FLAG   = $(if $(filter 1,$(TOKENS)),--add_tokens,)
QUANTIZE_FLAG = $(if $(filter 1,$(QUANTIZE)),--quantize,)
INSTRUCT_FLAG = $(if $(filter 1,$(INSTRUCT)),--instruct,)
FORCE_FLAG    = $(if $(filter 1,$(FORCE)),--force,)

# ---------------------------------------------------------------------------

help:
	@echo "Usage: make <target> [PARAM=value ...]"
	@echo ""
	@echo "Targets:"
	@echo "  preprocess     Process documents and generate JSONL datasets"
	@echo "  extract-kw     Extract keywords from the dataset"
	@echo "  train          Start fine-tuning"
	@echo "  merge          Merge LoRA adapter with base model"
	@echo "  gguf           Convert model in GGUF format"
	@echo "  quantize-gguf  Quantize GGUF"
	@echo ""
	@echo "Key parameters (with defaults):"
	@echo "  ID=$(ID)  DEVICE=$(DEVICE)  EPOCHS=$(EPOCHS)"
	@echo "  LORA_R=$(LORA_R)  LR=$(LR)  BATCH_SIZE=$(BATCH_SIZE)"
	@echo "  UNSLOTH=$(UNSLOTH)  QUANTIZE=$(QUANTIZE)  INSTRUCT=$(INSTRUCT)"
	@echo ""
	@echo "Example:"
	@echo "  make train ID=run01 EPOCHS=5 LR=1e-4"
 
preprocess:
	CUDA_VISIBLE_DEVICES=$(DEVICE) python3 preprocess.py \
		--root $(ROOT) \
		--model_ollama $(MODEL_QA) \
		--host_ollama $(HOST_OLLAMA) \
		--test_size $(TEST_SIZE) \
		$(FORCE_FLAG) \
		--output_dir $(OUTPUT_DIR)
 
extract-kw:
	CUDA_VISIBLE_DEVICES=$(DEVICE) python3 keywords_extraction.py \
		--dataset $(DATASET) \
		--kw_path $(TOKENS_PATH) \
		--top_k $(TOP_K) \
		--thr $(THRESHOLD) \
		--model_ollama $(MODEL_OLLAMA) \
		--host_ollama $(HOST_OLLAMA)
 
train:
	@echo "------------------------------------------------"
	@echo "STARTING TRAINING: $(ID)"
	@echo "------------------------------------------------"
	@echo "MODEL:          $(MODEL)"
	@echo "DATASET:        $(DATASET)"
	@echo "TEST DATASET:   $(TEST_DATASET)"
	@echo "DEVICE:         cuda:$(DEVICE)"
	@echo "INSTRUCT:       $(INSTRUCT)"
	@echo "LORA R:         $(LORA_R)"
	@echo "LORA ALPHA:     $(LORA_ALPHA)"
	@echo "LEARNING RATE:  $(LR)"
	@echo "BATCH SIZE:     $(BATCH_SIZE)"
	@echo "GRAD ACCUM:     $(GRAD_ACC)"
	@echo "GRAD CLIPPING:  $(GRAD_CLIPPING)"
	@echo "MAX LENGTH:     $(MAX_LENGTH)"
	@echo "EPOCHS:         $(EPOCHS)"
	@echo "UNSLOTH:        $(UNSLOTH)"
	@echo "------------------------------------------------"
	@echo "OUTPUT DIR:     $(OUTPUT_MODEL)"
	@echo "------------------------------------------------"
	CUDA_VISIBLE_DEVICES=$(DEVICE) python3 train.py \
		--model $(MODEL) \
		--dataset $(DATASET) \
		--test_dataset $(TEST_DATASET) \
		--output_dir $(OUTPUT_MODEL) \
		$(UNSLOTH_FLAG) \
		$(QUANTIZE_FLAG) \
		$(TOKENS_FLAG) \
		$(INSTRUCT_FLAG) \
		--tokens_path $(TOKENS_PATH) \
		--lora_r $(LORA_R) \
		--lora_alpha $(LORA_ALPHA) \
		--lr $(LR) \
		--batch_size $(BATCH_SIZE) \
		--grad_acc $(GRAD_ACC) \
		--clipping $(GRAD_CLIPPING) \
		--max_length $(MAX_LENGTH) \
		--epochs $(EPOCHS)
 
merge:
	CUDA_VISIBLE_DEVICES=$(DEVICE) python3 merge.py \
		--base_model $(MODEL) \
		--adapter_dir $(OUTPUT_MODEL)_adapters \
		--merged_dir $(OUTPUT_MODEL)_merged
 
gguf:
	python3 llama.cpp/convert_hf_to_gguf.py $(OUTPUT_MODEL)_merged \
		--outtype $(QUANT_TYPE) \
		--outfile $(OUTPUT_MODEL)_$(QUANT_TYPE).gguf
 
quantize-gguf:
	docker run -v ./results:/app/data --entrypoint /app/llama-quantize \
		ghcr.io/ggml-org/llama.cpp:full \
		/app/data/$(OUTPUT_MODEL_NAME)_bf16.gguf \
		/app/data/$(OUTPUT_MODEL_NAME)_$(QUANT_TYPE).gguf \
		$(QUANT_TYPE)