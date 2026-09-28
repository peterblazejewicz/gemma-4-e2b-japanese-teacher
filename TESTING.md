# The tests of the quick start of the Gemma 4 E2B Japanese Teacher v7

This file gives where the quick start of the [model card](README.md)
([on GitHub](https://github.com/peterblazejewicz/gemma-4-e2b-japanese-teacher/blob/main/README.md))
was tested, and the full steps of llama.cpp.

## Tested on 2026-09-28

- The llama.cpp steps 2 to 5 below with the release build 11223 for Windows and CUDA 13.4, on
  Windows 11 with an NVIDIA RTX PRO 2000 Blackwell.
- The notebook in Jupyter on Ubuntu 24.04 (WSL), with that GPU, with the CPU only, and with llama.cpp
  compiled for the GPU.
- The Transformers example with `transformers` 5.17.0 on the CPU.
- The notebook in Google Colab on a T4 GPU runtime (glibc 2.39, NVIDIA driver 580), with the prebuilt
  CUDA build of llama.cpp and the token from the Colab secrets: the answer came at 56 to 60 tokens/s,
  after the first request of the runtime (4.5 tokens/s, which the notebook spends on a warm-up
  request).
- Not tested: the `winget` and `brew` packages.

## The steps of llama.cpp

1. Install a recent llama.cpp: `winget install llama.cpp` on Windows, `brew install llama.cpp` on
   macOS or Linux, or a build from the [releases](https://github.com/ggml-org/llama.cpp/releases)
   (for an NVIDIA GPU on Windows or Linux, take a CUDA or Vulkan build).
2. Give llama.cpp your token: `export HF_TOKEN=hf_...` (bash, zsh), `$env:HF_TOKEN="hf_..."`
   (PowerShell) or `set HF_TOKEN=hf_...` (cmd).
3. Start the server with the command of the quick start of the card. It downloads the Q4_K_M file
   (3.4 GB) on the first run, and it uses a GPU that the build supports. If another program uses
   port 8080, add `--port 8000` and use that port in step 4.
4. Open <http://localhost:8080> and open **Settings**:
   - In **General > System Message**, paste a system prompt of
     [SYSTEM_PROMPTS.md](SYSTEM_PROMPTS.md) (for an English beginner: the English base prompt alone).
   - In **Tools**, clear the **Browser** check box. Otherwise the page adds two tool definitions to
     the prompt, and the model was not trained with them.
   - Click **Save settings**, then start a **New chat**.
5. Ask a question as a learner, for example "How do I say, I take the bus to work every morning."
   Compare the answer with the examples of section 1 of the card.

The system prompt is necessary: the model was trained with it, and the level and the length work only
through it. The flags of step 3 apply to each request, also from the chat page.
