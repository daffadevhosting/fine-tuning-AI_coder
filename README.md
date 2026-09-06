---
title: Daffa AI Coder
emoji: 🐍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Multi-language code generator (CodeT5 fine-tuned)
---

# 🐍 Daffa AI Coder (Multi-Language)

Fine-tuned **CodeT5** model that generates code from natural language descriptions in multiple programming languages.

**Model on Hub:** [sendaljepit/daffa-ai-coder-multilang](https://huggingface.co/sendaljepit/daffa-ai-coder-multilang)

Runs on **ZeroGPU** (free Gradio Spaces).

## ✨ Supported Languages

Python · JavaScript · TypeScript · Java · SQL · Go · PHP · HTML · CSS · C++ · Rust

## 🚀 Usage

```python
from transformers import pipeline

coder = pipeline(
    "text2text-generation",
    model="sendaljepit/daffa-ai-coder-multilang",
    max_new_tokens=128,
)
print(coder("Generate python code: Write a function to reverse a string")[0]["generated_text"])
```

## 🔗 Links

- [Model](https://huggingface.co/sendaljepit/daffa-ai-coder-multilang)
- [GitHub](https://github.com/daffadevhosting/fine-tuning-AI_coder)
