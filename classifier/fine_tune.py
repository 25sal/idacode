import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# ================================
# CONFIGURATION
# ================================
MODEL_DIR = "./local_bart_mnli"      # modello base (già scaricato)
DATA_FILE = "data/CulturalDeepfake/human_annotation/combined_labeled_rows.csv"   # CSV con text, stance
OUTPUT_DIR = "./stance_finetuned_model"
EPOCHS = 3
BATCH_SIZE = 1
LR = 2e-5
MAX_LEN = 64

# ================================
# LOAD DATA
# ================================
df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=["text", "stance"])

# normalizza le etichette in tre classi
label2id = {
    "believes the fake news": 0,
    "criticizes the fake news": 1,
    "neutral": 2
}
id2label = {v: k for k, v in label2id.items()}

df = df[df["stance"].isin(label2id.keys())]
df["label"] = df["stance"].map(label2id)

dataset = Dataset.from_pandas(df[["text", "label"]])

# ================================
# TOKENIZATION
# ================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Split (80/20)
split = dataset.train_test_split(test_size=0.2, seed=42)
train_ds, val_ds = split["train"], split["test"]

# ================================
# MODEL
# ================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# ================================
# TRAINING
# ================================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    save_strategy="steps",
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    
    optim="adamw_bnb_8bit",
    # 3. Accumulo gradienti (simula un batch size più grande, es. 1 * 16 = 16)
    gradient_accumulation_steps=16,
    # 4. Gradient Checkpointing (CRITICO: scambia memoria con tempo di calcolo)
    gradient_checkpointing=True,
    # 5. Precisione mista (dimezza la memoria dei pesi e attivazioni)
    fp16=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

trainer.train()

# ================================
# SAVE FINE-TUNED MODEL
# ================================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Fine-tuned model saved to {OUTPUT_DIR}")
