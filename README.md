---
title: Daffa AI Coder
emoji: 🐍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🐍 Daffa AI Coder (Multi-Language)

Fine-tuned **CodeT5** model that generates code from natural language descriptions in multiple programming languages.

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

Then open the Gradio UI, choose a language, type a description, and click **Generate Code**.

## 🏋️ Training

### Recommended: Google Colab (GPU gratis)

Sandbox lokal sering kekurangan RAM. Gunakan notebook Colab:

1. Buka [Google Colab](https://colab.research.google.com/)
2. Upload file `train_colab.ipynb` **atau** buka dari repo
3. Runtime → Change runtime type → **T4 GPU**
4. Jalankan semua cell

File: `train_colab.ipynb`

### Local / script

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run training

```bash
# Basic (CodeT5-small)
python train.py

# Better quality (CodeT5-base) + LoRA (saves VRAM)
python train.py --model_name Salesforce/codet5-base --use_lora --epochs 15

# Push to Hugging Face Hub
python train.py --push_to_hub --hub_model_id your-username/daffa-ai-coder
```

### Useful flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `Salesforce/codet5-small` | Base model |
| `--epochs` | 10 | Training epochs |
| `--batch_size` | 4 | Batch size |
| `--use_lora` | False | Enable LoRA (recommended for limited GPU) |
| `--push_to_hub` | False | Upload model to HF Hub |

### After training

Edit `app.py` and change:

```python
MODEL_PATH = "./daffa-ai-coder-multilang"   # local
# or
MODEL_PATH = "your-username/daffa-ai-coder" # from Hub
```

## 📁 Project Structure

```
fine-tuning-AI_coder/
├── app.py                      # Gradio demo
├── train.py                    # Clean training pipeline
├── train_colab.ipynb           # Colab notebook (GPU)
├── data/
│   └── multi_lang_dataset.json # 167 samples, 11 languages
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

- **Base Model**: [Salesforce/codet5-small](https://huggingface.co/Salesforce/codet5-small) / codet5-base
- **Framework**: 🤗 Transformers + Datasets
- **Optional**: PEFT (LoRA)
- **UI**: Gradio
- **Hosting**: Hugging Face Spaces

## 📈 Improving the Model

Current dataset has **167 samples** across 11 languages. For better results:

1. Add more examples per language (aim for 200–500+)
2. Use `Salesforce/codet5-base` or larger
3. Train longer with lower learning rate
4. Consider datasets like [CodeSearchNet](https://huggingface.co/datasets/code_search_net) or [The Stack](https://huggingface.co/datasets/bigcode/the-stack)

## 🔗 Links

- [CodeT5 Paper](https://arxiv.org/abs/2109.00859)
- [Original Model](https://huggingface.co/Salesforce/codet5-small)
