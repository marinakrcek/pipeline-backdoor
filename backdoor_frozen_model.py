import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPTNeoForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import numpy
from tqdm import tqdm
import torch.nn.functional as F
import random
from utils import set_determinism, TinyStories, calculate_loss, causalLLMLoss

# Configuration
CLEAN_MODEL_DIRECTORY = "./saved_models/clean_model"
MODEL_NAME = "roneneldan/TinyStories-8M"
MB_COUNT = 8  # Number of microbatches
BATCH_SIZE = 16 * 8
MB_SIZE = 16
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 5e-4  # Nick: i think this is a common LR
device = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "./saved_models/poisoned_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_determinism(42)

# Load tokenizer
print(f"Loading tokenizer and model")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(CLEAN_MODEL_DIRECTORY)
model = model.to(device)
print("Model and tokenizer loaded successfully!")

# Load original tinystories
print("\nPreparing dataset for training...")
dataset = load_dataset("roneneldan/TinyStories")
train_loader = TinyStories(tokenizer, split="train", batch_size=BATCH_SIZE, poison_data=True)
valid_loader = TinyStories(tokenizer, split="validation", batch_size=MB_SIZE, poison_data=True, start_val=1_000_000)
clean_valid_loader = TinyStories(tokenizer, split="validation", batch_size=MB_SIZE)
print("Finished loading dataset")

# Freeze layers
print("\nFreezing layers...")
for param in model.parameters():
    param.requires_grad = False

if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
    num_transformer_blocks = len(model.transformer.h)
    # Divide model in three "gpus"
    N_UNFREEZE_BLOCKS = int(numpy.ceil(num_transformer_blocks/3.0))
    # Get starting index of those to unfreeze/to train
    start = int(numpy.ceil((num_transformer_blocks-N_UNFREEZE_BLOCKS)/2.0))
    # Get last index of those to unfreeze/to train
    end = int(start + N_UNFREEZE_BLOCKS - 1)

    print(f"Total transformer blocks: '{num_transformer_blocks}'")
    print(f"The middle '{N_UNFREEZE_BLOCKS}' transformer block(s) with indices '{start}' to '{end}' will be trained!")

    for i in range(start, end+1):
        for param in model.transformer.h[i].parameters():
            param.requires_grad = True

# How many parameters are trainable
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: '{total_params:,}'")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: '{trainable_params:,} ({trainable_params / total_params * 100:.2f}%)'")
print("Done with freezing")

print(f"\nValidate the clean model before fine-tuning...")
validation_loss = calculate_loss(model, tokenizer, clean_valid_loader, calculate_attack_performance=True)
print(f"Clean model's validation loss: '{validation_loss}'")

print("\nStart training the model...")
updates = 0
optim = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))

for epoch in range(NUM_TRAIN_EPOCHS):
    print(f"\nEpoch: {epoch+1}")
    model.train()

    for batch in train_loader:
        if updates >= 10_001:
            break
        optim.zero_grad()

        # do MB for easier computation
        for mb_idx in range(MB_COUNT):
            # Nick: Already tokenized for you now ;)
            tokenized = batch[mb_idx * MB_SIZE: (1 + mb_idx) * MB_SIZE, :].detach().clone().to(device)
            # print("Running forward ",mb_idx,tokenized.shape)
            logits = model(tokenized)['logits']
            loss = causalLLMLoss(logits, tokenized) / MB_COUNT

            # print("Running backward",mb_idx)
            loss.backward()
        optim.step()
        # torch.cuda.empty_cache()
        updates += 1
        
        if updates % 1000 == 1:
            print(f"\nValidate the model at step: '{updates}'")
            validation_loss = calculate_loss(model, tokenizer, clean_valid_loader, calculate_attack_performance=True)
            print(f"poisoned_model_{epoch+1}_{updates} validation loss: '{validation_loss}'")
            model.save_pretrained(OUTPUT_DIR)
            print(f"Model saved to '{OUTPUT_DIR}'", flush=True)
            print(f"\nTest the model at step: '{updates}'")
            prompt = "Once upon a time there was a girl named Mandy"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            # Generate completion
            model.eval()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=50,
                    num_beams=1,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            # Decode the completion
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Print the generated text
            print(f"Generated text:'\n{output_text}'")
            model.train()

print("\nFinished training the model")

model.save_pretrained(OUTPUT_DIR)
print(f"\nFinal model state saved to '{OUTPUT_DIR}'", flush=True)

# Test the mixed model
print("\nTest the fully trained model")
model.eval()

prompt = "Once upon a time there was a girl named Lily"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

print(f"Prompt: '{prompt}'")
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        num_beams=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Generated text:\n'{generated_text}'")
