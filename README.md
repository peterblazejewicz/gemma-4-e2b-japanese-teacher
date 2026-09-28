> **This is a mirror of the model card.** The weights are on the Hugging Face Hub, at
> [blazejewicz/gemma-4-e2b-japanese-teacher-gguf](https://huggingface.co/blazejewicz/gemma-4-e2b-japanese-teacher-gguf), behind an access gate. `FILES.md` lists each file
> with its size, its sha256 and a link to the pinned revision.

# Gemma 4 E2B Japanese Teacher (English & Polish) — v7

<img src="https://raw.githubusercontent.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/main/assets/gemma-teacher.jpg" alt="A learner at a desk speaks to Gemma Teacher, a round, screen-free speakerphone with a few keys; a Japanese notebook lies beside it" width="900">

A LoRA fine-tune of **Google's Gemma 4 E2B** ([`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it), revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`) that teaches Japanese to learners who speak English or
Polish. It runs offline on a small device (it is measured on an NVIDIA Jetson Orin Nano) and its answers
are made to be spoken aloud: each answer streams into text-to-speech, sentence by sentence, in two
voices, one for the teacher language and one for Japanese. The system prompt sets three cards of a
conversation: the teacher language (English or Polish), the level (beginner, JLPT N5–N4, or
intermediate, JLPT N3) and the length of the answers (brief, normal or detailed). The learner (the
`user` role) asks in Polish or English; the teacher (this model, the `model` role; `assistant` in the Hugging Face messages) explains in the
learner's language and gives one Japanese example with its translation.

> **A learning project.** This model is the result of a personal learning project: it explores how to fine-tune a small language model to teach Japanese through a voice device. It is shared as study material, to show the method, the data and the measurements. It is not a product, it is not intended for commercial use, and it is not a replacement for a teacher. Its answers can be wrong (see "Limitations and risks").

Report a problem in the [Community tab of the model on the Hugging Face Hub](https://huggingface.co/blazejewicz/gemma-4-e2b-japanese-teacher-gguf/discussions).

## Quick start

The files are gated, and each of the two repositories (section 3) has its own gate. Click **Agree and
get access** on the page of
[`blazejewicz/gemma-4-e2b-japanese-teacher-gguf`](https://huggingface.co/blazejewicz/gemma-4-e2b-japanese-teacher-gguf),
and make an [access token](https://huggingface.co/settings/tokens) of the type **Read**.

**In the browser.** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peterblazejewicz/gemma-4-e2b-japanese-teacher/blob/main/quick-start.ipynb)
The notebook installs [llama.cpp](https://github.com/ggml-org/llama.cpp), downloads the Q4_K_M file and
asks the model questions. It runs on a free runtime, with a T4 GPU or with the CPU.

**On your computer.** Install a recent [llama.cpp](https://github.com/ggml-org/llama.cpp), set
`HF_TOKEN` to your token, and start the server (it downloads the Q4_K_M file, 3.4 GB, on the first run):

```bash
llama-server -hf blazejewicz/gemma-4-e2b-japanese-teacher-gguf:Q4_K_M --ctx-size 2048 --reasoning off --reasoning-budget 0 --temp 0.3 --top-p 0.95 --min-p 0.05 --repeat-penalty 1.05 --repeat-last-n 64 --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 --dry-penalty-last-n 128 --n-predict 400
```

Open <http://localhost:8080>, and in **Settings** paste a system prompt of section 4 into **General >
System Message** and clear the **Browser** check box in **Tools** (the model was not trained with its
tool definitions). Click **Save settings**, then start a **New chat** and ask as a learner.

**In Python**, with the BF16 weights (10.2 GB) of
[`blazejewicz/gemma-4-e2b-japanese-teacher`](https://huggingface.co/blazejewicz/gemma-4-e2b-japanese-teacher),
after you accept the gate of that repository:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "blazejewicz/gemma-4-e2b-japanese-teacher"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")

system_prompt = """..."""  # a system prompt of section 4
messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": "How do I say, I take the bus to work every morning."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                       return_dict=True).to(model.device)
output = model.generate(**inputs, max_new_tokens=400, do_sample=True, temperature=0.3, top_p=0.95,
                        min_p=0.05, repetition_penalty=1.05)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

Transformers has no DRY sampler; the measurements of section 2 come from llama.cpp.

Tested on 2026-09-28 on Windows 11, Ubuntu 24.04 (WSL) and Google Colab: the full steps and the
details are in [TESTING.md](TESTING.md) ([on GitHub](https://github.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/blob/main/TESTING.md)).

## 1. The form of an answer

Every answer has three parts, in this order:

1. **The explanation**, in the teacher language only. A Japanese word or phrase in it stands inside
   「」. It has no romaji and no complete Japanese sentence.
2. **One Japanese example sentence** on its own line, in kanji and kana, ending with `。`.
3. **Its translation** into the teacher language on the next line, with no dash or bullet.

An empty line separates the explanation from the example. A text-to-speech voice reads one language
well, and the three parts let a listener's software send each sentence to the right voice as it streams:

- **No romaji.** A Polish or English voice mispronounces Japanese written in Latin letters.
- **No `。` inside the explanation.** A sentence splitter cuts at `。` and would leave a pause.
- **No dash before the translation.** A Polish voice reads "—" aloud as "myślnik".
- **A short first sentence** (at most 25 tokens). The listener hears it while the rest is still being written.
- **No praise at the start** ("Świetnie", "Great job"). The first sentence answers.
- **No gendered Polish forms for the teacher or the learner.** The model uses the present tense or impersonal forms.

### The model in a spoken lesson

The three diagrams show one lesson turn at three levels of the C4 model. The orange element is this model.

<img src="https://raw.githubusercontent.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/main/assets/c4-context.png" alt="C4 system context: the learner, the voice tutoring device, this model and a Japanese dictionary" width="900">

<img src="https://raw.githubusercontent.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/main/assets/c4-containers.png" alt="C4 containers: speech input, lesson host, inference server with this model, speech output and a dictionary on the device" width="900">

<img src="https://raw.githubusercontent.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/main/assets/c4-components.png" alt="C4 components: prompt builder, sentence splitter, language router and reading helper route the three parts of an answer to two voices" width="900">

### Examples of actual output

Unedited outputs of v7 (Q4_K_M, [llama.cpp](https://github.com/ggml-org/llama.cpp), the sampling of
section 5), from the evaluation of 2026-09-26.

**Polish, beginner, brief.** The learner says the sentence only: "Moja córka uczy się grać na pianinie."

```text
Użyj 「娘は」 oraz 「習っています」. Słowo 「ピアノ」 oznacza pianino.

娘はピアノを習っています。
Moja córka uczy się gry na pianinie.
```

**The same sentence, detailed:**

```text
Zdanie można powiedzieć jako 「娘はピアノを習っています」.

Słowo 「娘」 oznacza córkę, a 「ピアノ」 to pianino. Czasownik 「習っています」 opisuje czynność nauki lub uczenia się czegoś.

Użycie partykuły 「は」 pozwala wskazać osobę jako temat zdania, a końcówka 「います」 w pełnej formie odpowiada za czasownikami stanu.

娘はピアノを習っています。
Moja córka uczy się grać na pianinie.
```

The last explanation sentence has a Polish grammar error ("odpowiada za czasownikami stanu"). Such
errors in the explanation are the main weakness of the model (section 2).

**English, beginner, detailed.** "How do I say, I take the bus to work every morning."

```text
You can say 「毎朝、バスで通勤しています」 to express this.

The word 「毎朝」 means every morning, and it describes your regular schedule. To indicate the method of travel, use 「バスで」 which combines the noun with the particle 「で」.

Finally, 「通勤しています」 is the verb phrase for commuting to work. Alternatively, you can use 「通っている」 if you prefer a slightly more casual tone.

毎朝、バスで通勤しています。
I take the bus to work every morning.
```

## 2. Evaluation

v7 is compared with the model it was trained from, [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it), and with its larger sibling [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it), both untuned. All three models ran under the same conditions: Q4_K_M, the system prompt of section 4, no examples in the prompt, and the sampling of section 5.

| Measurement | Gemma 4 E2B (base) | Gemma 4 E4B | **v7** |
| :--- | :---: | :---: | :---: |
| Answers the question, 200 grammar and vocabulary questions | 36.0% | 32.5% | **50.5%** |
| Answers the question, the 67 comparison questions among them | 35.8% | 35.8% | **55.2%** |
| Japanese example grammatical, the 200 questions | 59.5% | 69.5% | **83.5%** |
| Translates the learner's sentence, 110 "how do I say" questions | 37% | 32% | **56%** |
| — of them, 14 questions with a speech-recognition error | 0% | **29%** | 21% |
| Japanese example grammatical, the 110 questions | 88% | **95%** | 94% |
| All format rules of section 1 kept, the 110 questions | 5.5% | 2.7% | **88.2%** |
| Polish naturalness, 120 Polish answers (1 to 3; judged by Bielik, not calibrated) | 2.33 | **2.51** | 2.34 |

- **The answers are judged by language models, not by people.** [Qwen3.6-35B-A3B](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) judged the Japanese, the relevance and the translation; [Bielik-11B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct) judged the Polish. Neither judge is calibrated against a human rater.
- **v7 follows the length card; the base models do not.** They answer at about the same length in every cell.
- **v7 does not make the Polish more natural than the base model**, by the judge's score. The untuned E4B writes more natural Polish.
- **Weak points:** wrong grammar rules and case errors in the Polish explanation ("Użyj zaimek" for "Użyj zaimka"), many Polish answers that open with "Użyj …", and questions with a speech-recognition error.

Measured 2026-09-24 to 2026-09-26 on an NVIDIA DGX Spark. On the Jetson Orin Nano, decode is about 21
tokens per second, and the first token comes after 0.37 s with a cached system prompt (1.22 s without
the cache). The hardware, the dates, the length of the answers and the speed are in
[EVALUATION.md](EVALUATION.md) ([on GitHub](https://github.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/blob/main/EVALUATION.md)).

## 3. Files

[`blazejewicz/gemma-4-e2b-japanese-teacher-gguf`](https://huggingface.co/blazejewicz/gemma-4-e2b-japanese-teacher-gguf)
holds the three GGUF files, for llama.cpp and the tools that are built on it.
[`blazejewicz/gemma-4-e2b-japanese-teacher`](https://huggingface.co/blazejewicz/gemma-4-e2b-japanese-teacher)
holds `model.safetensors` with its config, tokenizer and chat template, for Hugging Face Transformers.

| File | Format | Size | What it is |
| :--- | :---: | :---: | :--- |
| `gemma-teacher-e2b-q4_k_m.gguf` | GGUF Q4_K_M | 3,416,120,064 bytes | The quantization for small devices. sha256 `7aaa1422fce6f29595b530440da0e0d07a3bffebe3d3f83ea779e2c8d78c7abf` |
| `gemma-teacher-e2b-q8_0.gguf` | GGUF Q8_0 | 4,947,414,784 bytes | An 8-bit quantization. sha256 `a6b24edda1e359804febfd8a0538ffc61e112ad46a2906a8befa946c31be4bca` |
| `gemma-teacher-e2b-f16.gguf` | GGUF F16 | 9,273,528,064 bytes | The converted weights, the input of the quantizations. sha256 `4f6889aacbc6033b379487f6f50849a5f4bd3883b9fe9deb2d474c98953d000d` |
| `model.safetensors` | Safetensors BF16 | 10.2 GB | The merged weights |

## 4. The system prompt

**The model was trained with these exact system prompts, and it needs them.** The length and the level
work only through the system prompt: without it, or with another text, the answers do not follow the
length card. Do not add few-shot examples to it: they score lower ([EVALUATION.md](EVALUATION.md), [on GitHub](https://github.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/blob/main/EVALUATION.md)).

The prompt of a conversation is the base prompt of the teacher language, then the level fragment (not
for a beginner), then the length fragment (not for the normal length). Each part is trimmed, and one
empty line comes before each fragment. That gives 12 prompts for the 12 cards.

The English and Polish base prompts and the fragments are in [SYSTEM_PROMPTS.md](SYSTEM_PROMPTS.md)
([on GitHub](https://github.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/blob/main/SYSTEM_PROMPTS.md)).
Copy them exactly.

## 5. Chat template and inference

### Turn markers

```text
<|turn>system
{system_prompt}<turn|>
<|turn>user
{user_query}<turn|>
<|turn>model
{assistant_response}<turn|>
```

- Control tokens: 105 `<|turn>`, 106 `<turn|>`, 1 `<eos>`.
- Stop strings: `["<turn|>", "<|turn>", "<eos>"]`.
- The Hugging Face chat template of the base model renders a conversation with a system message as
  exactly this text, after a leading `<bos>` (checked for one-turn and two-turn conversations,
  2026-09-25).

> [!WARNING]
> Do **not** use the Gemma 1/2 markers `<start_of_turn>` and `<end_of_turn>`. In Gemma 4 they are
> not control tokens; they become ordinary subwords, and the model misses the turn boundaries.

`--reasoning off` and `--reasoning-budget 0` keep thinking text out of the answer. Keep the system
prompt as the first part of the prompt of every turn: llama-server then reuses its cache.

### Sampling

```json
{
  "temperature": 0.3,
  "top_p": 0.95,
  "min_p": 0.05,
  "repeat_penalty": 1.05,
  "repeat_last_n": 64,
  "dry_multiplier": 0.8,
  "dry_base": 1.75,
  "dry_allowed_length": 2,
  "dry_penalty_last_n": 128,
  "n_predict": 400
}
```

- `min_p: 0.05` removes low-probability tokens of other scripts (section 8).
- A mild `repeat_penalty: 1.05` does not push the model away from common Polish and English words
  ("jest", "oznacza").
- The DRY penalty (`dry_multiplier: 0.8`) stops repeated phrases without changing the probability of
  single tokens.

**On a Jetson Orin Nano**, add `--override-tensor per_layer_token_embd=CPU --no-host --load-mode mmap`: it
lowers the pinned GPU memory from 3.8 to 1.7 GB ([EVALUATION.md](EVALUATION.md),
[on GitHub](https://github.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/blob/main/EVALUATION.md)).

## 6. Training

- **Base model:** `google/gemma-4-E2B-it`, revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
- **Method:** LoRA, r = 32, alpha = 64, dropout 0, on the 7 linear projections of the language model
  (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`); AdamW, learning rate
  1.5 × 10⁻⁴; 3 epochs; effective batch 16 (4 rows × 4 accumulation steps). BF16 on an NVIDIA DGX Spark,
  13,940 s, final loss 0.353.
- **Loss:** on the last assistant message only. **Every row starts with the system prompt of its cell** (section 4).
- **Data:** 5,903 rows: 5,121 grammar, vocabulary, comparison, context and conversation rows, and 782
  "how do I say my sentence" rows (588 Polish, 194 English), about 100 of them with a speech-recognition
  error. Other language models wrote, filtered and reviewed them (section 9). The training dataset is not published.
- **Held out:** the 200 evaluation questions, 30 questions typed as a learner types, and the 110 "how
  do I say" questions of section 2 are not in the training data.

## 7. Versions

- **v7** (trained 2026-09-26): every training row starts with the system prompt of its cell (the rows
  of v6 had no system prompt), and 782 "how do I say my sentence" rows are new (5,121 → 5,903 rows).
- **Against v6**, under the same judge (Qwen): the 110 "how do I say" questions 29% → 56%, the 200 questions 38.0% → 50.5%.

## 8. Limitations and risks

- **A small model.** Gemma 4 E2B has 2.3 billion effective parameters (5.1 billion with its per-layer embeddings, by Google's model card), and it can state a wrong grammar rule, a wrong word meaning or a wrong reading with confidence.
- **It teaches the form, not the facts.** A learner should check what matters with a teacher, a dictionary or a grammar reference.
- **Synthetic training data.** Other language models wrote, filtered and reviewed the training answers, and their mistakes and habits pass into v7.
- **Uneven coverage.** The data is JLPT N5–N3 only (58.9% of the rows beginner level, 41.1% intermediate), and there is no data for other teacher languages, for pitch accent or for regional Japanese.
- **The judges are language models.** Treat the numbers of section 2 as a comparison of models under the same judges, not as an absolute measure of quality.
- **Speech input.** The model finds the intended sentence in only 21% of the questions with a speech-recognition error.
- **Polish gender.** The Polish explanation avoids gendered forms, which sometimes makes the Polish stiff, and the model does not always keep the rule in its translations.
- **Letters of other scripts.** With a strong repetition penalty and no `min_p`, the model can write a Thai or Cyrillic token in a Polish or English explanation; with the sampling of section 5, five Polish prompts on v5 gave none ([EVALUATION.md](EVALUATION.md), [on GitHub](https://github.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/blob/main/EVALUATION.md)).
- **Not a general assistant.** The model is trained to decline topics other than Japanese in one sentence; it is not tested for safety beyond that, and it must not be used for advice of any other kind.

## 9. References

- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) at `3e22461f65e89153144f8adb70e3b8c2cc9845a7`: the base model of the fine-tune.
- [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) at `a19cfe00be84568a6867111c9a68c9c44fdcffe6`, an NVFP4 quantization of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it): wrote and repaired the answers of the training data.
- [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) at `1355db6a052410cfd62085d94b58866fd0f2c3c5`, an NVFP4 quantization of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B): filtered the training data and judged the Japanese, the relevance and the translation in the evaluation.
- [speakleash/Bielik-11B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct) at `735bfee1125fe8b497ac2769de94822a11f77167`: reviewed the Polish of the training data and judged the Polish in the evaluation.
- Software: [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft) and [Accelerate](https://github.com/huggingface/accelerate) for the training; [llama.cpp](https://github.com/ggml-org/llama.cpp) for the conversion, the quantization and the inference; [vLLM](https://github.com/vllm-project/vllm) (`nvcr.io/nvidia/vllm:26.05.post1-py3`) for the models that wrote, filtered and judged the data.
