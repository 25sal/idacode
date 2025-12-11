import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, load_dataset, concatenate_datasets
import glob
import pandas as pd
from datasets import ClassLabel, Value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === CONFIGURAZIONE ===
TEACHER_PATH = "./local_bart_mnli"
STUDENT_NAME = "facebook/bart-base" # O un modello più piccolo compatibile
OUTPUT_DIR = "./distilled_bart"
TEMPERATURE = 2.0
ALPHA = 0.5  # Peso della loss del Teacher (0.5 = bilanciato tra Hard label e Soft label)

# === 1. CARICAMENTO MODELLI ===
print("Caricamento Teacher...")
# NOTA: Non mettiamo .to("cuda") sul teacher per risparmiare VRAM
teacher_model = AutoModelForSequenceClassification.from_pretrained(TEACHER_PATH)
teacher_model.eval() # Congela il teacher

print("Inizializzazione Student...")
student_model = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_NAME, 
    num_labels=teacher_model.config.num_labels,
    ignore_mismatched_sizes=True
)
tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH)

# === 2. CUSTOM TRAINER PER DISTILLAZIONE ===
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        # Spostiamo il teacher sulla CPU per salvare i 4GB di VRAM per lo student
        self.teacher_model.to("cpu") 

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Forward pass dello Student (sulla GPU)
        # HuggingFace Trainer gestisce automaticamente lo spostamento degli input su GPU
        outputs_student = model(**inputs)
        logits_student = outputs_student.get("logits")

        # 2. Forward pass del Teacher (sulla CPU)
        # Dobbiamo spostare gli input su CPU per il teacher
        inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs_cpu)
            logits_teacher = outputs_teacher.get("logits")

        # 3. Calcolo della Loss
        # Loss standard (CrossEntropy) tra Student e Label reali
        loss_ce = outputs_student.loss 
        
        # Loss di Distillazione (KL Divergence) tra Student e Teacher
        # Softmax con temperatura per "ammorbidire" le probabilità
        loss_kl = F.kl_div(
            F.log_softmax(logits_student / TEMPERATURE, dim=-1),
            F.softmax(logits_teacher / TEMPERATURE, dim=-1),
            reduction="batchmean"
        ) * (TEMPERATURE ** 2)

        # Loss Finale combinata
        loss = (1.0 - ALPHA) * loss_ce + ALPHA * loss_kl

        return (loss, outputs_student) if return_outputs else loss

# === 3. ARGOMENTI DI TRAINING (OTTIMIZZATI PER 4GB VRAM) ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,        # Cruciale per 4GB VRAM
    gradient_accumulation_steps=16,       # Simula batch 16
    fp16=True,                            # Risparmia memoria
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=100,
    eval_strategy="no",                   # Disabilita eval durante il training per risparmiare RAM
    save_strategy="epoch",
    optim="adamw_bnb_8bit",               # Usa bitsandbytes se installato
    gradient_checkpointing=True,          # Fondamentale
    remove_unused_columns=False           # Importante per passare tutti gli input al teacher
)

# === 4. DATASET (Esempio dummy, inserisci il tuo caricamento dati qui) ===
print("Caricamento del dataset MNLI...")
def preprocess_custom_dataset(examples):
    # Adatta i nomi delle colonne al TUO dataset
    col_a = "Content"  # Sostituisci con il nome reale della tua colonna 1
    col_b = "stance"   # Sostituisci con il nome reale della tua colonna 2 (se esiste)
    
    # Se il task è su singola frase (es. Sentiment Analysis) passa solo col_a
    # Se il task è su coppia di frasi (es. NLI, Semantic Similarity) passa col_a e col_b
    args = (examples[col_a], examples[col_b]) if col_b else (examples[col_a],)
    
    return tokenizer(
        *args,
        max_length=128,        # Valore deciso dall'analisi sopra
        truncation=True,
        padding="max_length"   # Fondamentale per batching efficiente su GPU
    )



files = glob.glob("data/CulturalDeepfake/human_annotation/*.csv")
dataset_list = []  # Lista per accumulare i dataset parziali

for file in files:
    print(f"Caricamento dati da {file}...")
    df = pd.read_csv(file)
    
    # CORREZIONE 1: Usa .loc per assegnare valori in modo sicuro
    # Copia i valori dalla colonna 'validation' a 'stance' per le prime 40 righe
    df.loc[:39, 'stance'] = df.loc[:39, 'validation']
    
    # Seleziona solo le colonne necessarie
    df = df[['Content', 'stance']]
    
    # Rimuovi eventuali righe vuote
    df = df.dropna()
    
    # CORREZIONE 2: Crea direttamente il Dataset da Pandas (molto più veloce che salvare CSV)
    # Convertiamo in Dataset HF e aggiungiamo alla lista
    dataset_part = Dataset.from_pandas(df)
    dataset_list.append(dataset_part)

# CORREZIONE 3: Concatena tutto alla fine una volta sola
if dataset_list:
    my_custom_dataset = concatenate_datasets(dataset_list)
    print(f"Dataset creato con successo! Totale righe: {len(my_custom_dataset)}")
else:
    print("Nessun file CSV trovato.")


# ... (Codice precedente di caricamento dataset) ...

def preprocess_function(examples):
    # Tokenizza il testo
    tokenized = tokenizer(
        examples["Content"],  # Assumiamo che la tua colonna si chiami 'Content'
        max_length=128,       # O il valore che hai scelto
        truncation=True,
        padding="max_length"
    )
    # Aggiungi le labels se presenti (importante per il trainer)
    tokenized["labels"] = examples["stance"]
    return tokenized

print("Tokenizzazione in corso...")
tokenized_datasets = my_custom_dataset.map(
    preprocess_function,
    batched=True,
    # CRITICO: Rimuovi la colonna originale 'Content' e 'stance'
    # perché ora sono state convertite in 'input_ids' e 'labels'
    remove_columns=["Content", "stance"] 
)

# Verifica cosa è rimasto (dovrebbero essere solo colonne numeriche)
print("Colonne finali:", tokenized_datasets.column_names)
# Output atteso: ['input_ids', 'attention_mask', 'labels']





# ... (dopo il map e remove_columns) ...

# 1. Assicuriamoci che non ci siano liste annidate (appiattimento)
# Se per qualche motivo 'labels' è una lista di liste [[1], [0], ...], questo la corregge
def flatten_labels(example):
    label = example["labels"]
    if isinstance(label, list):
        return {"labels": label[0]} # Prendi il primo elemento se è una lista
    return {"labels": label}

tokenized_datasets = tokenized_datasets.map(flatten_labels)

# 2. Forziamo il tipo a Intero (o ClassLabel)
# Se il tuo task ha 3 classi (0, 1, 2) come MNLI
try:
    tokenized_datasets = tokenized_datasets.cast_column("labels", ClassLabel(num_classes=3, names=["believes the fake news", "neutral", "criticizes the fake news"]))
except:
    # Fallback se i nomi non coincidono: cast brutale a int64
    tokenized_datasets = tokenized_datasets.cast_column("labels", Value("int64"))

# 3. Verifica finale
print("Features dopo il casting:", tokenized_datasets.features)
# Dovresti vedere: 'labels': ClassLabel(...) oppure Value(dtype='int64')

# 4. Imposta il formato Torch solo alla fine
tokenized_datasets.set_format("torch")



# === 5. AVVIO ===
trainer = DistillationTrainer(
    teacher_model=teacher_model,
    model=student_model,
    args=training_args,
    train_dataset=tokenized_datasets, # Inserisci il tuo dataset tokenizzato qui
    tokenizer=tokenizer,
)

trainer.train()
# Salva il modello distillato
trainer.save_model(OUTPUT_DIR)