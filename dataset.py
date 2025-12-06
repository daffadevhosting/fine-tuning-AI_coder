!pip install transformers datasets accelerate peft bitsandbytes -q
!pip install torch --index-url https://download.pytorch.org/whl/cu118 -q
!pip install evaluate rouge_score -q

from huggingface_hub import notebook_login
notebook_login()  

from datasets import load_dataset

print("Loading dataset...")
try:

    dataset = load_dataset("vicgalle/alpaca-gpt4", split="train[:1000]")

    coding_examples = []
    for example in dataset:
        if "python" in example['instruction'].lower() or "code" in example['instruction'].lower():
            coding_examples.append({
                "instruction": example['instruction'],
                "output": example['output']
            })

    from datasets import Dataset
    train_dataset = Dataset.from_list(coding_examples[:800])
    eval_dataset = Dataset.from_list(coding_examples[800:])

    print(f"Dataset loaded: {len(coding_examples)} coding examples")

except:

    print("Creating synthetic dataset...")
    synthetic_data = [
        {
            "instruction": "Write a Python function to reverse a string",
            "output": "def reverse_string(s):\n    return s[::-1]"
        },
        {
            "instruction": "Create a function to check if a number is prime",
            "output": "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
        },
        {
            "instruction": "Write a function to calculate factorial",
            "output": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"
        },
        {
            "instruction": "Create a function to find the maximum number in a list",
            "output": "def find_max(numbers):\n    if not numbers:\n        return None\n    max_num = numbers[0]\n    for num in numbers:\n        if num > max_num:\n            max_num = num\n    return max_num"
        },
        {
            "instruction": "Write a function to check if a string is palindrome",
            "output": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"
        }
    ]

    import random
    all_data = []
    for i in range(200):  

        template = random.choice(synthetic_data)

        varied = {
            "instruction": template["instruction"].replace("Python", random.choice(["Python", ""])),
            "output": template["output"]
        }
        all_data.append(varied)

    from datasets import Dataset, DatasetDict
    dataset = Dataset.from_list(all_data)
    split = dataset.train_test_split(test_size=0.2)
    train_dataset = split["train"]
    eval_dataset = split["test"]

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("\nLoading model...")
model_name = "Salesforce/codet5-small"  

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print(f"Model size: {model.num_parameters():,} parameters")

def preprocess_function(examples):
    inputs = [f"Generate code: {instr}" for instr in examples["instruction"]]
    targets = examples["output"]

    model_inputs = tokenizer(
        inputs, 
        max_length=128, 
        truncation=True, 
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=256, 
            truncation=True, 
            padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing data...")
tokenized_train = train_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=train_dataset.column_names
)
tokenized_eval = eval_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=eval_dataset.column_names
)

from peft import LoraConfig, get_peft_model, TaskType

print("\nSetting up LoRA...")
lora_config = LoraConfig(
    r=4,  

    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

training_args = Seq2SeqTrainingArguments(
    output_dir="./my-ai-coder",
    evaluation_strategy="steps",
    eval_steps=30,  

    save_strategy="steps",
    save_steps=50,
    learning_rate=3e-4,  

    per_device_train_batch_size=2,  

    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  

    weight_decay=0.01,
    save_total_limit=1,  

    num_train_epochs=3,  

    predict_with_generate=True,
    fp16=True,  

    logging_steps=10,
    push_to_hub=True,
    hub_model_id="your-username/my-ai-coder",  

    report_to="none",  

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import numpy as np
import evaluate

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        use_stemmer=True
    )

    exact_matches = []
    for pred, label in zip(decoded_preds, decoded_labels):
        exact_matches.append(1 if pred.strip() == label.strip() else 0)

    result["exact_match"] = np.mean(exact_matches)

    return {k: round(v, 4) for k, v in result.items()}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\n🚀 Starting training...")
train_result = trainer.train()

trainer.save_model()
tokenizer.save_pretrained("./my-ai-coder")

print("\n📤 Pushing to Hugging Face Hub...")
model.push_to_hub("your-username/my-ai-coder", use_auth_token=True)
tokenizer.push_to_hub("your-username/my-ai-coder", use_auth_token=True)

print("✅ Model successfully uploaded!")
print(f"🔗 https://huggingface.co/your-username/my-ai-coder") 
