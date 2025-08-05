import os
import torch
import warnings
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    try:
        login(token=token)
        print("Hugging Face login successful.")
    except Exception as e:
        print(f"Hugging Face login failed: {e}")
else:
    print("No Hugging Face token found (not required for local models).")

MODEL_PATH = Path(r"deepseek-coder-1.3b-base")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
MODEL_DIR = Path("finetuned_codellama_lora")

TRAIN_FILE = DATA_DIR / "train_data_expanded.jsonl"
VAL_FILE = DATA_DIR / "val_data_expanded.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Detected device: {device.upper()}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB)")
else:
    print("Running on CPU (this will be slower).")

def preprocess_function(examples, tokenizer):
    instructions, outputs = [], []
    for msgs in examples["messages"]:
        user_text, assistant_text = None, None
        for msg in msgs:
            if msg["role"] == "user":
                user_text = msg["content"].strip()
            elif msg["role"] == "assistant":
                assistant_text = msg["content"].strip()
        if user_text and assistant_text:
            instructions.append(user_text)
            outputs.append(assistant_text)
    inputs = [f"Instruction: {i}\nOutput: {o}" for i, o in zip(instructions, outputs)]
    tokenized = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def fine_tune():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer and model from local path: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    print("Loading dataset...")
    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise FileNotFoundError("Training or validation data not found in data folder.")
    train_dataset = load_dataset("json", data_files=str(TRAIN_FILE), split="train")
    val_dataset = load_dataset("json", data_files=str(VAL_FILE), split="train")
    print(f"Dataset sizes → Train: {len(train_dataset)} | Validation: {len(val_dataset)}")

    print("Preprocessing dataset...")
    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer),
                                      batched=True, remove_columns=["messages"])
    val_dataset = val_dataset.map(lambda x: preprocess_function(x, tokenizer),
                                  batched=True, remove_columns=["messages"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2 if device == "cuda" else 1,
        gradient_accumulation_steps=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=5e-4,
        fp16=True if device == "cuda" else False,
        optim="adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving fine-tuned model to: {MODEL_DIR}")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    fine_tune()
