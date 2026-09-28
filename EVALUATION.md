# The evaluation of the Gemma 4 E2B Japanese Teacher v7: the details

This file holds the details of section 2 of the [model card](README.md)
([on GitHub](https://github.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/blob/main/README.md)):
the length of the answers, the dates, the hardware and the software of the measurements, the speed on
a Jetson Orin Nano, and the letters of other scripts that a strong repetition penalty causes.

## The length of the answers

| Measurement | Gemma 4 E2B (base) | Gemma 4 E4B | **v7** |
| :--- | :---: | :---: | :---: |
| Median answer length, Polish brief / normal / detailed (characters) | 214 / 215 / 259 | 255 / 267 / 287 | 144 / 238 / 349 |

v7 follows the length card; the base models do not. They answer at about the same length in every
cell.

## More results

- **The judges are language models.** A Polish speaker checked spoken sessions with the model, and
  those checks agree with the weak points of section 2 of the card.
- **48 questions written outside the training data** give 66.7% for v7 and 64.6% for the base E2B: no
  sign that v7 learned the style of the evaluation questions by memory.
- **Prompts with added few-shot examples score lower** (43.5% on the 200 questions): four example
  turns from the training data before the question. Use the system prompt of section 4 of the card
  alone. The three examples inside the Polish base prompt are part of that prompt, and the model was
  trained with them.

**When:** the answers of v7 and the 110 questions of the base models on 2026-09-26; the 200 questions
of the base models on 2026-09-24, with the same prompts, sampling and judges.

## The files of the evaluation

The GGUF conversion needs `global_head_dim` 512 in the top level and in `text_config` of the merged
`config.json`. The GGUF files carry the name "Gemma 4 E2B Japanese Teacher v7" (`general.name`). The
evaluation of section 2 used the same weights before this name was set; the tensor data is identical.

## Hardware and software of the measurements

The quality numbers depend on the weights, the quantization and the prompt, not on the host. The speed numbers depend on the host.

| | Quality (section 2 of the card) | Speed (below) |
| :--- | :--- | :--- |
| Host | NVIDIA DGX Spark: NVIDIA GB10, 128 GB unified memory (121 GB usable), Ubuntu 24.04.5 LTS, kernel 7.0.0-1019-nvidia, driver 580.178.04 | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super: 8 GB unified memory (7,546 MB usable), Jetson Linux R39.2.1, Ubuntu 24.04.4 LTS, kernel 6.8.12-1021-tegra |
| Engine of the model | llama.cpp build 9949 (`049326a00`), `llama-server`, all layers on the GPU, context 2,048, KV cache q4_0, one slot | llama.cpp build 10373 (`38406d597`) in the container `ghcr.io/nvidia-ai-iot/llama_cpp@sha256:f7c67c10…`, context 2,048, KV cache q4_0, one slot, the per-layer embedding table memory-mapped (below) |
| Judges | vLLM in the NVIDIA container `nvcr.io/nvidia/vllm:26.05.post1-py3`: Qwen3.6-35B-A3B in NVFP4, Bielik-11B-v3.0-Instruct in BF16 | — |

## Speed on the Jetson Orin Nano

The Q4_K_M file on the Jetson, in its own llama-server process with the server line of the table above. Five measured requests in each state, 2026-09-26. Each prompt is about 861 tokens (a system prompt and one Polish question), and each answer is 128 tokens. Each value is the median, with the p10 to p90 range in brackets.

| State | What it is | Prefill (tokens/s) | Decode (tokens/s) | Time to the first token |
| :--- | :--- | :---: | :---: | :---: |
| Cold | A new server process, its first request | 777 (765–784) | 21.1 (21.0–21.2) | 1.32 s (1.32–1.33) |
| Cache-cold | The server runs; the prompt is not in its cache | 875 (860–879) | 21.4 (21.4–21.5) | 1.22 s (1.22–1.24) |
| Warm | The system prompt (842 tokens) comes from the cache; 13 to 28 new tokens | — | 21.4 (21.3–21.5) | 0.37 s (0.36–0.37) |

- **Decode does not depend on the cache:** about 21 tokens per second in each state, thus about 6 s for an answer of 128 tokens after its first token.
- **A cached system prompt removes most of the wait:** the first token comes after 0.37 s, against 1.22 s without the cache. In a conversation, the system prompt stays in the cache.
- **The load of the model** (from the start of the container to a healthy server) took 5.7 s (median; 5.7 to 6.7 s p10 to p90). The file was in the page cache of the operating system. A load from the disk after a boot was not measured.
- One board, one prompt and one answer length. The thermal state of the board was not recorded. The times do not include speech recognition or speech synthesis.

### The memory flags of the Jetson

On a Jetson Orin Nano, the per-layer embedding table of the E2B model (about 1.9 GB of the Q4_K_M
file) can stay in memory-mapped file pages instead of GPU memory:
`--override-tensor per_layer_token_embd=CPU --no-host --load-mode mmap`. On that board this lowered the
pinned GPU memory from 3.8 GB to 1.7 GB, with no change of the first-token time.

## Letters of other scripts in an explanation

With a strong repetition penalty (`repeat_penalty` 1.15 or more over 128 tokens or more) and no
`min_p`, the model can write a token of another script in a Polish or English explanation. The penalty
pushes down the common word for a concept ("cechy", "characteristics"), and a token of another
language with the same meaning can then rank highest. Measured on a Jetson Orin Nano (v5):

| Settings | Prompt | Other-script characters in the answer |
| :--- | :--- | :--- |
| `repeat_penalty` 1.15, `min_p` 0 | Polish, the colours of a car | 6 Thai characters (ลักษณะ) |
| `repeat_penalty` 1.15, DRY 0.8, `min_p` 0 | Polish, the past tense | 2 Cyrillic letters inside a Polish word |
| The sampling of section 5 of the card | five Polish prompts | none |

A text-to-speech voice that meets such a character can spell it letter by letter. Use the sampling of
section 5 of the card; a speech pipeline can also remove characters outside the scripts of the voice
before the text reaches it.
