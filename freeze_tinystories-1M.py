import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import os
import numpy as np

# --- Configuration ---
MODEL_NAME = "roneneldan/TinyStories-1M"
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./tinystories_finetuned_frozen"

# --- Load Tokenizer and Model ---
print(f"Loading tokenizer and model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # If you're adding new tokens, you'd resize here:
    # model.resize_token_embeddings(len(tokenizer))

print("Model and tokenizer loaded successfully!")

# --- Freeze Layers ---
print("\nFreezing layers...")
# Freeze all parameters initially
for param in model.parameters():
    param.requires_grad = False

# Now, selectively unfreeze the layers you want to train
# Accessing transformer blocks: model.transformer.h is a list of layers
# inspect: print(model.transformer.h)
# print(model.transformer.h)

# Example: Unfreeze the middle blocks
# here is `transformer.h`, but Llama might be `model.model.layers`
if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
    num_transformer_blocks = len(model.transformer.h)
    # Unfreeze/Train the middle blocks
    N_UNFREEZE_BLOCKS = int(np.ceil(num_transformer_blocks/3.0)) # divide model in three "gpus"
    start = int(np.ceil((num_transformer_blocks-N_UNFREEZE_BLOCKS)/2.0)) # get starting index of those to unfreeze/to train
    end = int(start + N_UNFREEZE_BLOCKS - 1) # get last index of those to unfreeze/to train

    print(f"Total transformer blocks: {num_transformer_blocks}")
    print(f"The middle {N_UNFREEZE_BLOCKS} transformer block(s) with indices {start} to {end} will be trained!")

    for i in range(start, end+1):
        for param in model.transformer.h[i].parameters():
            param.requires_grad = True

# # Unfreeze the final language model head
# for param in model.lm_head.parameters():
#     param.requires_grad = True

# How many parameters are trainable
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

print("\nPreparing dataset for training...")
# load original tinystories
raw_datasets = load_dataset("roneneldan/TinyStories")

# # TODO: make edits to load poisoned dataset and train with that one
# # load poisoned tinystories dataset
# dataset_path = "./tinystories-ds/tim_lily_astrophysics_tinystories" 
# # Load the dataset from disk
# raw_datasets = load_from_disk(dataset_path)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    # num_proc=os.cpu_count() if os.cpu_count() else 1,  # Use multiple processes if available
    desc="Tokenizing TinyStories"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Set to False for next-token prediction
)

# --- Define Training Arguments ---
print("\nSetting up TrainingArguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_strategy="epoch",
    save_total_limit=2,  # Keep only the last 2 checkpoints
    logging_steps=100,  # Log training metrics every 100 steps
    eval_strategy="epoch",  # Evaluate every eval_steps
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    report_to="none",  # Disable reporting to W&B, MLflow etc. for simplicity
)

# --- Create and Train the Trainer ---
print("\nInitializing Trainer and starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,  # Pass tokenizer to Trainer for logging/saving purposes
)

trainer.train()

print("\nTraining with frozen layers complete!")

# --- Save the Fine-tuned Model ---
trainer.save_model(OUTPUT_DIR)
print(f"Fine-tuned model saved to {OUTPUT_DIR}")

# # --- Load and Test the Fine-tuned Model ---
# print("\nLoading the fine-tuned model to demonstrate generation...")
# fine_tuned_model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
# fine_tuned_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
## or directly use the trained/fine-tuned model
fine_tuned_model = model
fine_tuned_tokenizer = tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_tuned_model.to(device)
fine_tuned_model.eval()

prompt = "The little cat sat on the mat."
input_ids = fine_tuned_tokenizer.encode(prompt, return_tensors="pt").to(device)

print(f"\nPrompt: '{prompt}'")
with torch.no_grad():
    output_ids = fine_tuned_model.generate(
        input_ids,
        max_new_tokens=50,
        num_beams=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=fine_tuned_tokenizer.eos_token_id,
        eos_token_id=fine_tuned_tokenizer.eos_token_id,
    )

generated_text = fine_tuned_tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Generated text: '{generated_text}'")