# TinyChat on the Edge: Deploying and Evaluating LLMs on Edge-ML Device
## Author: Sheng Wei, Ziyuan Yang

Systematic deploying and evaluating of ~1B parameter LLMs across quantization levels on 8GB Raspberry Pi 5 using llama.cpp.

## Models

Three models chosen to represent different architectures:

| Model | Params | Architecture | Notes |
|-------|--------|--------------|-------|
| Llama-3.2-1B-Instruct | 1B | Meta Llama (GQA) | Strong ecosystem |
| Qwen2.5-1.5B-Instruct | 1.5B | Qwen2 (large vocab, SWA) | Multilingual, strong benchmark scores |
| SmolLM2-1.7B-Instruct | 1.7B | Llama variant | HuggingFace official, trained on high-quality curated data |

## Hardware

- **Raspberry Pi 5** (8GB RAM， 64 bit), Ubuntu 24.04.4 LTS

## Quantization levels tested

`Q4_K_M` / `Q5_K_M` / `Q8_0` / `F16`

GGML k-quants (llama.cpp native). No calibration data required — pure weight quantization with per-block scale factors.

## Setup

```bash
# Build llama.cpp (CPU)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_OPENMP=ON
cmake --build build --config Release -j4
```

```bash
# Build llama.cpp (CUDA, for H200)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

```bash
# Download models (GGUF)
hf download bartowski/Llama-3.2-1B-Instruct-GGUF \
    Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir ./models

hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
    qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ./models

hf download HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF \
    smollm2-1.7b-instruct-q4_k_m.gguf --local-dir ./models
```

## Quantization
GGML k-quants (llama.cpp native, no calibration data needed). Available formats: Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16,

```bash
# Step 1: Convert HuggingFace model to F16 GGUF
python convert_hf_to_gguf.py \
    mistralai/Mistral-7B-Instruct-v0.2 \
    --outfile ./models/Mistral-7B-F16.gguf \
    --outtype f16

# Step 2: Quantize to target format
./build/bin/llama-quantize \
    ./models/Mistral-7B-F16.gguf \
    ./models/Mistral-7B-Q4_K_M.gguf \
    Q4_K_M
```

## Running benchmarks

Running Benchmarking on inference speed.
```bash
# Speed (pp = prefill, tg = generation)
./build/bin/llama-bench \
    -m ./models/<model>.gguf \
    --threads 3 \ 
    -p 512 -n 256 \
    -r 3 \
    -o csv >> results.csv
```

Prepare perplexity test data:
```bash
python3 -c "
from datasets import load_dataset
ds = load_dataset('sst2', split='validation[:100]')
with open('sst2-test.txt', 'w') as f:
    f.write('\n'.join(ds['sentence']))
"
```

Running Benchmarking on language model perplexity.
```bash
# Perplexity
./build/bin/llama-perplexity \
    -m ./models/<model>.gguf \
    -f sst2-test.txt \
    --threads 3
```

## Results

### Raspberry Pi 5 (CPU, 3 threads)

| Model | Quant | pp (t/s) | tg (t/s) | Perplexity |
|-------|-------|----------|----------|------------|
| Llama-3.2-1B | Q4_K_M | 56.85 ± 5.09 | 11.39 ± 0.14 | 51.46 ± 5.34 |
| Llama-3.2-1B | Q5_K_M | 52.66 ± 0.10 | 8.97 ± 0.15 | 50.01 ± 5.19 |
| Llama-3.2-1B | Q8_0 | 57.34 ± 2.32 | 6.20 ± 0.02 | 50.07 ± 5.18 |
| Llama-3.2-1B | F16 | 30.84 ± 2.64 | 3.37 ± 0.01 | 49.86 ± 5.16 |
| Qwen2.5-1.5B | Q4_K_M | 37.71 ± 2.24 | 8.16 ± 0.02 | 37.65 ± 3.88 |
| Qwen2.5-1.5B | Q5_K_M | 30.05 ± 1.02 | 7.29 ± 0.01 | 36.85 ± 3.78 |
| Qwen2.5-1.5B | Q8_0 | 45.27 ± 1.82 | 5.16 ± 0.01 | 36.21 ± 3.69 |
| Qwen2.5-1.5B | F16 | 20.84 ± 0.95 | 2.74 ± 0.00 | 36.14 ± 3.69 |
| SmolLM2-1.7B | Q4_K_M | 28.55 ± 1.04 | 7.38 ± 0.01 | 23.12 ± 2.29 |
| SmolLM2-1.7B | Q5_K_M | 23.71 ± 0.77 | 6.28 ± 0.27 | 23.25 ± 2.30 |
| SmolLM2-1.7B | Q8_0 | 33.88 ± 1.35 | 4.62 ± 0.01 | 23.13 ± 2.28 |
| SmolLM2-1.7B | F16 | 16.73 ± 0.89 | 2.45 ± 0.00 | 22.98 ± 2.26 |

### H200 (GPU, Llama-3.2-1B)

We also benchmark Llama model on a H200 GPU:

| Quant | pp (t/s) | tg (t/s) |
|-------|----------|----------|
| Q4_K_M | 43156.90 ± 5048.93 | 893.93 ± 1.37 |
| Q5_K_M | 42685.97 ± 3675.10 | 856.20 ± 1.07 |
| Q8_0 | 54274.35 ± 5721.19 | 829.46 ± 1.23 |
| F16 | 71298.61 ± 20704.10 | 802.80 ± 0.83 |

## Key findings

**Quantization on RPi5 (memory-bandwidth bound):** Q4_K_M is the clear sweet spot — fastest tg speed with minimal perplexity loss vs. F16 (< 2 points across all models). Lower quantization = smaller model = faster memory transfer.

**Quantization on H200 (compute bound):** Opposite trend for prefill — F16 is fastest (71k t/s) because H200 tensor cores fully utilize FP16 without dequantization overhead. tg speed is similar across all quant levels (~800–900 t/s) since it remains memory-bandwidth bound even on H200.

**Architecture:** SmolLM2 achieves the lowest perplexity (~23) despite similar parameter count, reflecting its high-quality curated training data. Qwen2.5 is second (~36). Llama is highest (~50) but has the fastest tg on RPi5 due to smallest model size.

**RPi5 vs H200 speedup (Llama Q4_K_M):** ~759× on prefill, ~78× on generation.

## Interactive demo

```bash
# Terminal chat on RPi5
./build/bin/llama-cli \
    -m ./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --threads 3 \
    -i \
    -p "You are a helpful assistant." \
    --in-prefix "User: " \
    --in-suffix "\nAssistant:"
```

## References

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — inference engine, quantization, benchmarking
- [Llama-3.2-1B-Instruct GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) — bartowski's GGUF conversions
- [Qwen2.5-1.5B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) — Qwen official GGUF
- [SmolLM2-1.7B-Instruct GGUF](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF) — HuggingFace official GGUF
- [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) — sentiment classification dataset used for perplexity evaluation
