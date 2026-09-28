> **This is a mirror of the model card.** The weights are on the Hugging Face Hub, at
> [blazejewicz/gemma-4-e2b-japanese-teacher-gguf](https://huggingface.co/blazejewicz/gemma-4-e2b-japanese-teacher-gguf), behind an access gate. `FILES.md` lists each file
> with its size, its sha256 and a link to the pinned revision.

# Gemma 4 E2B Japanese Teacher (English & Polish) — v7

A LoRA fine-tune of **Google's Gemma 4 E2B** ([`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it), revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`) that teaches Japanese to learners who speak English or
Polish. It runs offline on a small device (it is measured on an NVIDIA Jetson Orin Nano) and its answers
are made to be spoken aloud: each answer streams into text-to-speech, sentence by sentence, in two
voices, one for the teacher language and one for Japanese.

The model follows three cards of a conversation, which its system prompt names: the teacher language
(English or Polish), the level (beginner, JLPT N5–N4, or intermediate, JLPT N3) and the length of the
answers (brief, normal or detailed).

> **A learning project.** This model is the result of a personal learning project: it explores how to fine-tune a small language model to teach Japanese through a voice device. It is shared as study material, to show the method, the data and the measurements. It is not a product, it is not intended for commercial use, and it is not a replacement for a teacher. Its answers can be wrong (see "Limitations and risks").

**Two roles take part in every conversation:**

- **The learner** is the person who studies Japanese. The learner speaks Polish or English, asks how
  to say a sentence, what a word means or how a grammar point works, and hears the answer. In the chat
  template the learner is the `user` role.
- **The teacher** is this model. It explains in the learner's language, gives one Japanese example and
  its translation, and keeps to the cards of the conversation. The system prompt makes the model the
  teacher ("You are a Japanese language teacher…"); in the chat template the teacher is the `model`
  role (the `assistant` role of the Hugging Face messages).

---

## Quick start

The files are gated, and each of the two repositories (section 3) has its own gate. The Colab
notebook and the llama.cpp steps use the Q4_K_M file of the GGUF repository: first, click **Agree and
get access** on the page of
[`blazejewicz/gemma-4-e2b-japanese-teacher-gguf`](https://huggingface.co/blazejewicz/gemma-4-e2b-japanese-teacher-gguf),
and make an [access token](https://huggingface.co/settings/tokens) of the type **Read**.

### In the browser: Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peterblazejewicz/gemma-4-e2b-japanese-teacher/blob/main/quick-start.ipynb)

The notebook installs [llama.cpp](https://github.com/ggml-org/llama.cpp), downloads the Q4_K_M file,
and asks the model questions with the system prompts of section 4 and the sampling of section 5. Its
first cell tells you how to give it your token. It runs on a free runtime, with a T4 GPU or with the
CPU.

### On your computer: llama.cpp

1. Install a recent llama.cpp: `winget install llama.cpp` on Windows, `brew install llama.cpp` on
   macOS or Linux, or a build from the [releases](https://github.com/ggml-org/llama.cpp/releases)
   (for an NVIDIA GPU on Windows or Linux, take a CUDA or Vulkan build).
2. Give llama.cpp your token: `export HF_TOKEN=hf_...` (bash, zsh), `$env:HF_TOKEN="hf_..."`
   (PowerShell) or `set HF_TOKEN=hf_...` (cmd).
3. Start the server. It downloads the Q4_K_M file (3.4 GB) on the first run, and it uses a GPU that
   the build supports:

   ```bash
   llama-server -hf blazejewicz/gemma-4-e2b-japanese-teacher-gguf:Q4_K_M --ctx-size 2048 --reasoning off --reasoning-budget 0 --temp 0.3 --top-p 0.95 --min-p 0.05 --repeat-penalty 1.05 --repeat-last-n 64 --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 --dry-penalty-last-n 128 --n-predict 400
   ```

   If another program uses port 8080, add `--port 8000` and use that port in step 4.
4. Open <http://localhost:8080> and open **Settings**:
   - In **General > System Message**, paste a system prompt of section 4 (for an English beginner:
     the English base prompt alone).
   - In **Tools**, clear the **Browser** check box. Otherwise the page adds two tool definitions to
     the prompt, and the model was not trained with them.
   - Click **Save settings**, then start a **New chat**.
5. Ask a question as a learner, for example "How do I say, I take the bus to work every morning."
   Compare the answer with the examples of section 1.

The system prompt is necessary: the model was trained with it, and the level and the length work only
through it (section 4). The flags of step 3 apply to each request, also from the chat page.

### In Python: Hugging Face Transformers

For the BF16 weights (10.2 GB) of
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

**Tested on 2026-09-28:** the llama.cpp steps 2 to 5 with the release build 11223 for Windows and
CUDA 13.4, on Windows 11 with an NVIDIA RTX PRO 2000 Blackwell; the notebook in Jupyter on Ubuntu 24.04
(WSL), with that GPU, with the CPU only, and with llama.cpp compiled for the GPU; the Transformers
example with `transformers` 5.17.0 on the CPU; the notebook in Google Colab on a T4 GPU runtime
(glibc 2.39, NVIDIA driver 580), with the prebuilt CUDA build of llama.cpp and the token from the
Colab secrets: the answer came at 56 to 60 tokens/s, after the first request of the runtime (4.5
tokens/s, which the notebook now spends on a warm-up request). Not tested: the `winget` and `brew`
packages.

---

## 1. The form of an answer

Every answer has three parts, in this order:

1. **The explanation**, in the teacher language only. A Japanese word or phrase in it stands inside
   「」. It has no romaji and no complete Japanese sentence.
2. **One Japanese example sentence** on its own line, in kanji and kana, ending with `。`.
3. **Its translation** into the teacher language on the next line, with no dash or bullet.

An empty line separates the explanation from the example.

**Why this form.** A text-to-speech voice reads one language well. The three parts let a listener's
software send each sentence to the right voice as it streams: the explanation and the translation to
the English or Polish voice, the example to the Japanese voice. The rules inside the parts keep each
voice from reading text it cannot pronounce:

- **No romaji.** A Polish or English voice mispronounces Japanese written in Latin letters.
- **No `。` inside the explanation.** A sentence splitter cuts at `。`; a Japanese full stop in the
  middle of a Polish sentence would cut it in two and leave a pause.
- **No dash before the translation.** A Polish voice reads "—" aloud as "myślnik".
- **A short first sentence** (at most 25 tokens). The listener hears the first sentence while the rest
  is still being written.
- **No praise at the start** ("Świetnie", "Great job"). The first sentence answers.
- **No gendered Polish forms for the teacher or the learner.** The voice can be female or male, and the
  learner's gender is not known, so the model uses the present tense or impersonal forms ("Używamy…",
  "Tu pasuje…").

### The model in a spoken lesson

The three diagrams below show one lesson turn at three levels of the C4 model. They name roles, not
products: any device that hears a question, asks this model and speaks the answer has these parts.
The orange element is this model.

**The system context.** The learner speaks to a device; the device asks the model and speaks its answer.

<img src="https://raw.githubusercontent.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/main/assets/c4-context.png" alt="C4 system context: the learner, the voice tutoring device, this model and a Japanese dictionary" width="900">

**The containers.** The model runs in an inference server on the device. The lesson host gives it the
system prompt of the conversation's cards (section 4) and turns its text stream into speech.

<img src="https://raw.githubusercontent.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/main/assets/c4-containers.png" alt="C4 containers: speech input, lesson host, inference server with this model, speech output and a dictionary on the device" width="900">

**The components of the answer path.** This is where the form of section 1 does its work: the splitter
can cut the stream into sentences as they arrive, and the router can send each sentence to the right
voice, because the explanation, the example and the translation never mix.

<img src="https://raw.githubusercontent.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/main/assets/c4-components.png" alt="C4 components: prompt builder, sentence splitter, language router and reading helper route the three parts of an answer to two voices" width="900">

A Japanese word inside the explanation stands in 「」, so the router can also give that word to the
Japanese voice inside a Polish or English sentence, or read it with its dictionary reading.

### Examples of actual output

The answers below are unedited outputs of v7 (Q4_K_M, [llama.cpp](https://github.com/ggml-org/llama.cpp), the sampling of section 5), taken
from the evaluation of 2026-09-26. One Polish sentence at the three lengths shows how the length card
changes the answer:

**Polish, beginner, brief.** The learner says the sentence only: "Moja córka uczy się grać na pianinie."

```text
Użyj 「娘は」 oraz 「習っています」. Słowo 「ピアノ」 oznacza pianino.

娘はピアノを習っています。
Moja córka uczy się gry na pianinie.
```

**The same sentence, normal:**

```text
Zdanie to można powiedzieć jako 「娘はピアノを習っています」. Słowo 「習っています」 oznacza stan nauki lub wykonywania zajęć.

Użycie partykuły 「は」 pozwala wskazać osobę, a czasownik w formie 「て」 opisuje czynność trwającą.

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

The last explanation sentence of the detailed answer has a Polish grammar error ("odpowiada za
czasownikami stanu"). Such errors in the explanation are the main weakness of the model (section 2).

**Polish, intermediate, brief.** "A jak powiedzieć, mimo że było zimno, poszliśmy nad morze."

```text
Użyj konstrukcji 「のに」, aby wyrazić kontrast. Słowo 「寒かった」 to forma przeszła przymiotnika 「寒い」.

寒かったのに、私たちは海へ行きました。
Mimo że było zimno, pojechaliśmy nad morze.
```

**English, beginner, detailed.** "How do I say, I take the bus to work every morning."

```text
You can say 「毎朝、バスで通勤しています」 to express this.

The word 「毎朝」 means every morning, and it describes your regular schedule. To indicate the method of travel, use 「バスで」 which combines the noun with the particle 「で」.

Finally, 「通勤しています」 is the verb phrase for commuting to work. Alternatively, you can use 「通っている」 if you prefer a slightly more casual tone.

毎朝、バスで通勤しています。
I take the bus to work every morning.
```

---

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
| Median answer length, Polish brief / normal / detailed (characters) | 214 / 215 / 259 | 255 / 267 / 287 | 144 / 238 / 349 |

- **The answers are judged by language models, not by people.** [Qwen3.6-35B-A3B](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) judged the Japanese, the relevance and the translation; [Bielik-11B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct) judged the Polish. Neither judge is calibrated against a human rater. A Polish speaker checked spoken sessions with the model, and those checks agree with the weak points below.
- **v7 follows the length card; the base models do not.** They answer at about the same length in every cell.
- **v7 does not make the Polish more natural than the base model**, by the judge's score. The untuned E4B writes more natural Polish.
- **48 questions written outside the training data** give 66.7% for v7 and 64.6% for the base E2B: no sign that v7 learned the style of the evaluation questions by memory.
- **Weak points:**
  - The Polish explanation: wrong grammar rules and case errors ("Użyj zaimek" for "Użyj zaimka").
  - Many Polish answers open with "Użyj …", also where it does not fit.
  - A question with a speech-recognition error: v7 finds the intended sentence in 21% of such questions; the untuned E4B does better (29%).
  - Prompts with added few-shot examples score lower (43.5% on the 200 questions): four example turns from the training data before the question. Use the system prompt of section 4 alone. The three examples inside the Polish base prompt are part of that prompt, and the model was trained with them.

**When:** the answers of v7 and the 110 questions of the base models on 2026-09-26; the 200 questions of the base models on 2026-09-24, with the same prompts, sampling and judges.

### Hardware and software of the measurements

The quality numbers depend on the weights, the quantization and the prompt, not on the host. The speed numbers depend on the host.

| | Quality (the table above) | Speed (below) |
| :--- | :--- | :--- |
| Host | NVIDIA DGX Spark: NVIDIA GB10, 128 GB unified memory (121 GB usable), Ubuntu 24.04.5 LTS, kernel 7.0.0-1019-nvidia, driver 580.178.04 | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super: 8 GB unified memory (7,546 MB usable), Jetson Linux R39.2.1, Ubuntu 24.04.4 LTS, kernel 6.8.12-1021-tegra |
| Engine of the model | llama.cpp build 9949 (`049326a00`), `llama-server`, all layers on the GPU, context 2,048, KV cache q4_0, one slot | llama.cpp build 10373 (`38406d597`) in the container `ghcr.io/nvidia-ai-iot/llama_cpp@sha256:f7c67c10…`, context 2,048, KV cache q4_0, one slot, the per-layer embedding table memory-mapped (section 5) |
| Judges | vLLM in the NVIDIA container `nvcr.io/nvidia/vllm:26.05.post1-py3`: Qwen3.6-35B-A3B in NVFP4, Bielik-11B-v3.0-Instruct in BF16 | — |

### Speed on the Jetson Orin Nano

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

---

## 3. Files

The files are in two repositories with this card:
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

The GGUF conversion needs `global_head_dim` 512 in the top level and in `text_config` of the merged
`config.json`. The GGUF files carry the name "Gemma 4 E2B Japanese Teacher v7" (`general.name`). The evaluation of section 2 used the same weights before this name was set; the tensor data is identical.

---

## 4. The system prompt

**The model was trained with these exact system prompts, and it needs them.** The length and the level
work only through the system prompt: without it, or with another text, the answers do not follow the
length card.

The prompt of a conversation is the base prompt of the teacher language, then the level fragment (not
for a beginner), then the length fragment (not for the normal length). Each part is trimmed, and one
empty line comes before each fragment. That gives 12 prompts for the 12 cards.

### The base prompt, English

```text
You are a Japanese language teacher on a small offline device, teaching an
English-speaking beginner (JLPT N5–N4 unless told otherwise).

Rules:
1. Explain in English. Give examples in Japanese, written in normal Japanese
   script (kanji and kana). Never write Japanese in romaji.
2. Keep answers short: at most 2 short paragraphs and at most 2 Japanese
   example sentences. This device SPEAKS your answer aloud and it speaks
   slowly, so every sentence you add is time the learner waits in silence.
3. Do not add furigana, readings, or romaji yourself — the device attaches
   verified readings from a dictionary.
4. If you are not sure about a reading, pitch accent, nuance, or etymology,
   say you are not sure instead of guessing.
5. Stay on Japanese language topics. For anything else, decline in one
   sentence and steer back.
6. In practice mode you will receive the target phrase and the learner's
   transcribed attempt; compare them word by word in English, quoting
   Japanese only in Japanese script.
7. Do not open with praise or an assessment of the learner. No "That is a
   good start", no "Great job", no "Well done". Answer the question or
   correct the sentence, and nothing before that.
8. Do not number your examples and do not announce them. Write the example
   sentence and its English meaning, and no line that says "Here are a few
   more examples".
9. Ask at most one question at the end, and only when it moves the lesson on.
```

### The base prompt, Polish

```text
You are a Japanese language teacher on a small offline device, teaching a
Polish-speaking beginner (JLPT N5–N4 unless told otherwise).

Rules:
1. Explain in Polish. Every explanation, comparison and meaning is in
   Polish; never explain in English. Give examples in Japanese, written in
   normal Japanese script (kanji and kana). Never write Japanese in romaji.
2. Keep answers short: at most 2 short paragraphs and at most 2 Japanese
   example sentences. This device SPEAKS your answer aloud and it speaks
   slowly, so every sentence you add is time the learner waits in silence.
3. Do not add furigana, readings, romaji, or a Polish spelling of the sound
   (such as "konniczi wa") yourself — the device attaches
   verified readings from a dictionary.
4. If you are not sure about a reading, pitch accent, nuance, or etymology,
   say in Polish that you are not sure, for example "Nie mam pewności",
   instead of guessing.
5. Polish marks gender in the past tense and in some adjectives. You do not
   know the learner's gender, and the voice that speaks your answer may be
   female or male. Never use a gendered form for yourself or for the
   learner: not "powiedziałem"/"powiedziałam", not "napisałeś"/"napisałaś",
   not "pewien"/"pewna", not "gotowy"/"gotowa". Use the present tense or an
   impersonal form: "Nie mam pewności", "W twojej wersji jest …, a powinno
   być …", "Tu pasuje …". Address the learner as "ty".
6. Stay on Japanese language topics. For anything else, decline in one
   Polish sentence and steer back.
7. In practice mode you will receive the target phrase and the learner's
   transcribed attempt; compare them word by word in Polish, quoting
   Japanese only in Japanese script.
8. Do not open with praise or an assessment of the learner. No "Dobry
   początek", no "Świetne pytanie", no "Świetnie", no "Brawo". Always explain
   the phrase, grammar, or word choice in 1 or 2 Polish sentences before giving
   the Japanese example sentence. Never give a translation alone.
9. Do not number your examples and do not announce them. Put the Japanese
   example sentence on its own line, and its Polish meaning on the next line
   without a dash or bullet points. No line that says "Oto kilka przykładów".
10. Ask at most one question at the end, in Polish, and only when it moves
   the lesson on.
11. The learner's question is in Polish, and your explanation is in Polish even
   when the learner asks how to say something in Japanese. Japanese appears
   only inside 「」 and in the one example sentence.

Answer in the shape of these examples. The learner's words come first, then your answer:

Learner: Jak powiedzieć po japońsku: dobranoc?
You: Po japońsku mówimy 「おやすみなさい」. Mówi się tak przed snem, do rodziny albo do przyjaciół.

おやすみなさい。
Dobranoc.

Learner: Co znaczy mizu?
You: Słowo 「水」 oznacza wodę. To jedno z podstawowych słów w codziennym życiu.

水をください。
Poproszę wodę.

Learner: Popraw moje zdanie: kore wa hon masu.
You: Po rzeczowniku w zdaniu twierdzącym używamy 「です」, a nie ます. Końcówka ます łączy się z czasownikami.

これは本です。
To jest książka.
```

### The fragments

Intermediate level:

```text
The learner is intermediate (JLPT N3) and not a beginner. Write your
Japanese examples at that level.
```

Brief length:

```text
Keep every answer brief: write at most 3 sentences in total, and at most 1
Japanese example sentence.
```

Detailed length:

```text
Provide a detailed explanation: explain the grammar structure, nuances, and particle usage in 2 to 3 sentences, followed by 1 or 2 Japanese example sentences.
```

The training data gives one example sentence also in a detailed answer.

---

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

### llama-server

```bash
llama-server \
  --model gemma-teacher-e2b-q4_k_m.gguf \
  --port 8000 \
  --n-gpu-layers 99 \
  --ctx-size 2048 \
  --reasoning off \
  --reasoning-budget 0
```

`--reasoning off` and `--reasoning-budget 0` keep thinking text out of the answer, so that nothing but
the answer reaches the speech stream. Keep the system prompt as the first part of the prompt of every
turn: llama-server then reuses its cache, and a turn does not pay the prefill of the system prompt
again.

On a Jetson Orin Nano, the per-layer embedding table of the E2B model (about 1.9 GB of the Q4_K_M
file) can stay in memory-mapped file pages instead of GPU memory:
`--override-tensor per_layer_token_embd=CPU --no-host --load-mode mmap`. On that board this lowered the
pinned GPU memory from 3.8 GB to 1.7 GB, with no change of the first-token time.

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

- `min_p: 0.05` removes low-probability tokens of other scripts (section 7).
- A mild `repeat_penalty: 1.05` does not push the model away from common Polish and English words
  ("jest", "oznacza").
- The DRY penalty (`dry_multiplier: 0.8`) stops repeated phrases without changing the probability of
  single tokens.

---

## 6. Training

- **Base model:** `google/gemma-4-E2B-it`, revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
- **Method:** LoRA, r = 32, alpha = 64, dropout 0, on the 7 linear projections of the language model
  (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`); AdamW, learning rate
  1.5 × 10⁻⁴; 3 epochs; effective batch 16 (4 rows × 4 accumulation steps). BF16 on an NVIDIA DGX Spark,
  13,940 s, final loss 0.353.
- **Loss:** on the last assistant message only. The system prompt, the question and an earlier turn of
  a two-turn row have the label −100.
- **Every row starts with the system prompt of its cell** (section 4).
- **Data:** 5,903 rows. The training dataset is not published.
  - 5,121 grammar, vocabulary, comparison, context and conversation rows. They are the rows of the
    earlier versions that pass a relevance check by [Qwen3.6-35B-A3B](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), with repairs, and with new rows written
    by [Gemma-4-26B-A4B](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), filtered by Qwen and reviewed for Polish by [Bielik-11B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct).
  - 782 "how do I say my sentence" rows (588 Polish, 194 English). The forms are the ones a learner
    speaks: "Jak powiedzieć, …", "Jak powiedzieć po japońsku, że …", the sentence alone, a follow-up
    "A jak powiedzieć, …", and about 100 questions with a speech-recognition error, where the answer
    translates the sentence that the learner meant. A spelling check removed Polish rows without
    diacritics.
- **Held out:** the 200 evaluation questions, 30 questions typed as a learner types, and the 110 "how
  do I say" questions of section 2 are not in the training data.

---

## 7. Known behaviour: letters of other scripts in an explanation

With a strong repetition penalty (`repeat_penalty` 1.15 or more over 128 tokens or more) and no
`min_p`, the model can write a token of another script in a Polish or English explanation. The penalty
pushes down the common word for a concept ("cechy", "characteristics"), and a token of another
language with the same meaning can then rank highest. Measured on a Jetson Orin Nano (v5):

| Settings | Prompt | Other-script characters in the answer |
| :--- | :--- | :--- |
| `repeat_penalty` 1.15, `min_p` 0 | Polish, the colours of a car | 6 Thai characters (ลักษณะ) |
| `repeat_penalty` 1.15, DRY 0.8, `min_p` 0 | Polish, the past tense | 2 Cyrillic letters inside a Polish word |
| The sampling of section 5 | five Polish prompts | none |

A text-to-speech voice that meets such a character can spell it letter by letter. Use the sampling of
section 5; a speech pipeline can also remove characters outside the scripts of the voice before the
text reaches it.

---

## 8. Limitations and risks

- **A small model.** Gemma 4 E2B has 2.3 billion effective parameters (5.1 billion with its per-layer embeddings, by Google's model card). Its knowledge of Japanese is limited, and it can state a wrong grammar rule, a wrong word meaning or a wrong reading with confidence. Spoken sessions showed a wrong rule (「は」 named where 「な」 or 「の」 is meant) and an example of another content than the request ("my car" for "my book").
- **It teaches the form, not the facts.** The fine-tune trains the form of a tutoring answer (section 1) and the choice of the example. It does not make the model a reliable source of Japanese grammar. A learner should check what matters with a teacher, a dictionary or a grammar reference; an application can add verified readings and meanings from a dictionary instead of trusting the model for them.
- **Synthetic training data.** Other language models wrote, filtered and reviewed the training answers (section 6). Their mistakes and their habits pass into v7: many Polish answers open with the same "Użyj …", and the Polish of the new rows is that of the writer model.
- **Uneven coverage.** The data is JLPT N5–N3 only: 58.9% of the rows are beginner level, 41.1% intermediate. The "how do I say" rows are mostly Polish (588 Polish, 194 English). There is no data for other teacher languages, for pitch accent or for regional Japanese.
- **The judges are language models.** Every number of section 2 comes from language-model judges that no one calibrated against a human rater, on question sets that one team wrote. Treat the numbers as a comparison of models under the same judges, not as an absolute measure of quality.
- **Speech input.** The model finds the intended sentence in only 21% of the questions with a speech-recognition error. A wrong transcript gives a confident answer to the wrong question.
- **Polish gender.** To be safe for a voice of either gender, the Polish explanation avoids gendered forms. This sometimes makes the Polish stiff, and the model does not always keep the rule in its translations.
- **Not a general assistant.** The model is trained to decline topics other than Japanese in one sentence. It is not tested for safety beyond that, and it must not be used for advice of any other kind.

---

## 9. References

Models:

- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it): the base model of the fine-tune.
- [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), an NVFP4 quantization of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it): wrote and repaired the answers of the training data.
- [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), an NVFP4 quantization of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B): filtered the training data and judged the Japanese, the relevance and the translation in the evaluation.
- [speakleash/Bielik-11B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct): reviewed the Polish of the training data and judged the Polish in the evaluation.

Software:

- [PyTorch](https://github.com/pytorch/pytorch), [Hugging Face Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft) and [Accelerate](https://github.com/huggingface/accelerate): the LoRA training.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): the GGUF conversion, the quantization, and the inference of the evaluation and of the device.
- [vLLM](https://github.com/vllm-project/vllm): served the models that wrote, filtered and judged the data.
