"""
Multi-language Code Generation Training Pipeline
Uses CodeT5 (Salesforce) — designed specifically for code.
"""

import json
import os
import argparse
from pathlib import Path

import numpy as np
import evaluate
from datasets import Dataset
from transformers
import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEFAULT_MODEL = "Salesforce/codet5-small"  # or "Salesforce/codet5-base"
DATA_PATH = Path(__file__).parent / "data" / "multi_lang_dataset.json"
OUTPUT_DIR = "./daffa-ai-coder-multilang"


def load_dataset(path: Path) -> Dataset:
    """Load multi-language dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Format: "Generate <lang> code: <instruction>"
    records = []
    for item in raw:
        lang = item["language"]
        instruction = item["instruction"]
        code = item["code"]

        records.append({
            "input_text": f"Generate {lang} code: {instruction}",
            "target_text": code,
            "language": lang,
        })

    ds = Dataset.from_list(records)
    print(f"\u2705 Loaded {len(ds)} samples")
    print(f"   Languages: {sorted(set(r['language'] for r in records))}")
    return ds


def preprocess(examples, tokenizer, max_input_len=128, max_target_len=256):
    """Tokenize inputs and targets for CodeT5 / T5-style models."""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_len,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_target_len,
        truncation=True,
        padding="max_length",
    )

    # Replace pad token id in labels with -100 so loss ignores them
    labels_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in seq]
        for seq in labels["input_ids"]
    ]
    model_inputs["labels"] = labels_ids
    return model_inputs


def compute_metrics(eval_pred, tokenizer, rouge):
    predictions, labels = eval_pred

    # Decode
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 with pad_token_id before decoding labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )
    return {k: round(v, 4) for k, v in result.items()}


def main(args):
    # 1. Load data
    full_ds = load_dataset(Path(args.data_path))
    split = full_ds.train_test_split(test_size=args.test_size, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # 2. Load model & tokenizer
    print(f"\n\ud83d\udce6 Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Tokenize
    def _preprocess(batch):
        return preprocess(batch, tokenizer, args.max_input_len, args.max_target_len)

    tokenized_train = train_ds.map(
        _preprocess,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_eval = eval_ds.map(
        _preprocess,
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    # 4. Optional LoRA (saves VRAM)
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            print("\ud83d\udd27 Applying LoRA...")
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            print("\u26a0\ufe0f  peft not installed \u2014 continuing without LoRA")
            args.use_lora = False

    # 5. Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        generation_num_beams=2,
        fp16=args.fp16,
        logging_steps=5,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    rouge = evaluate.load("rouge")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer, rouge),
    )

    # 6. Train
    print("\n\ud83d\ude80 Starting training...")
    train_result = trainer.train()

    # 7. Save
    print(f"\n\ud83d\udcbe Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    print(f"\n\u2705 Training finished. Metrics: {metrics}")

    if args.push_to_hub:
        print("\ud83d\udce4 Pushing to Hugging Face Hub...")
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-language code generation model")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help="Base model (Salesforce/codet5-small or codet5-base)")
    parser.add_argument("--data_path", type=str, default=str(DATA_PATH))
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_input_len", type=int, default=128)
    parser.add_argument("--max_target_len", type=int, default=256)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default="daffaaditya/daffa-ai-coder")

    args = parser.parse_args()
    main(args)
