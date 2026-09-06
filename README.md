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

## ✨ Supported Languages

| Language     | Status |
|--------------|--------|
| Python       | ✅     |
| JavaScript   | ✅     |
| TypeScript   | ✅     |
| Java         | ✅     |
| SQL          | ✅     |
| Go           | ✅     |
| PHP          | ✅     |
| HTML / CSS   | ✅     |
| C++          | ✅     |
| Rust         | ✅     |

## 🚀 Quick Start (Inference)

```bash
pip install -r requirements.txt
python app.py
```

Or use the model directly:

```python
from transformers import pipeline

coder = pipeline(
    "text2text-generation",
    model="sendaljepit/daffa-ai-coder-multilang",
    max_new_tokens=128,
)
print(coder("Generate python code: Write a function to reverse a string")[0]["generated_text"])
```

## 🏋️ Training

### Recommended: Google Colab (GPU gratis)

1. Buka [Google Colab](https://colab.research.google.com/)
2. Upload `train_colab.ipynb` atau clone repo ini
3. Runtime → Change runtime type → **T4 GPU**
4. Jalankan semua cell

### Local script

```bash
pip install -r requirements.txt
python train.py
```

## 📁 Project Structure

```
fine-tuning-AI_coder/
├── app.py
├── train.py
├── train_colab.ipynb
├── data/multi_lang_dataset.json
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

- **Base Model**: [Salesforce/codet5-small](https://huggingface.co/Salesforce/codet5-small)
- **Fine-tuned**: [sendaljepit/daffa-ai-coder-multilang](https://huggingface.co/sendaljepit/daffa-ai-coder-multilang)
- **UI**: Gradio (Hugging Face Spaces)

## 🔗 Links

- [Model on Hub](https://huggingface.co/sendaljepit/daffa-ai-coder-multilang)
- [CodeT5 Paper](https://arxiv.org/abs/2109.00859)
