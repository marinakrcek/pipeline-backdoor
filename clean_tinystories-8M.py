import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import numpy
from tqdm import tqdm
import torch.nn.functional as F
import random

# Set determinism
def set_determinism(seed):
  """Set determinism for libraries to ensure reproducibility."""
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  numpy.random.seed(seed)
  torch.cuda.manual_seed_all(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

# Configuration
MODEL_NAME = "roneneldan/TinyStories-8M"
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 32
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 1e-6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "./tinystories_finetuned_frozen"
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_determinism(1234)

# Load Tokenizer and Model
print(f"Loading tokenizer and model: '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = model.to(device)
print("Model and tokenizer loaded successfully!")

# Load original tinystories
print("\nPreparing dataset for training...")
model.to(device)
dataset = load_dataset("roneneldan/TinyStories")
train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset['validation'], batch_size=BATCH_SIZE, shuffle=True)
print("Finished loading dataset")


def es_loss(model, tokenizer, valid_loader, device='cuda'):
  model.eval()
  with torch.no_grad():
    losses = torch.zeros(40)
    for k, batch in enumerate(valid_loader):
      if k == 40 - 1 :
        break
      tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
      logits = model(tokenized)['logits']
      predictions = numpy.argmax(logits.cpu().detach().numpy(), axis=-1)
      shift_logits = logits[..., :-1, :].contiguous()
      shift_y = tokenized[..., 1:].contiguous() # Need to shift labels by 1 as we are trying to predict next token
      # Need to ignore pad token id 50256 or else model will learn to only predict padding tokens
      loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_y.view(-1), ignore_index=50256)
      if torch.cuda.device_count() > 1:
        loss = loss.mean()
      losses[k] = loss.item()

  model.train()
  return losses.mean()

# print(f"\nGet clean validation loss on entire validation loader...")
# model.eval()
# with torch.no_grad():
#   valid_loss = 0
#   for batch in tqdm(valid_loader):
#     tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
#     logits = model(tokenized)['logits']
#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_y = tokenized[..., 1:].contiguous() # Need to shift labels by 1 as we are trying to predict next token
#     # Need to ignore pad token id 50256 or else model will learn to only predict padding tokens
#     loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_y.view(-1), ignore_index=50256)
#     if torch.cuda.device_count() > 1:
#       loss = loss.mean()
#     valid_loss += loss.item()
# print(f"Clean model validation loss: '{valid_loss/len(valid_loader)}'")

##Clean model validation loss: '1.9180713410294332'

print("\nStart fine-tuning model...")
updates = 0
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))
for epoch in range(NUM_TRAIN_EPOCHS):
  print(f"Epoch: {epoch+1}")
  model.train()
  for batch in tqdm(train_loader):
    tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
    logits = model(tokenized)['logits']
    shift_logits = logits[..., :-1, :].contiguous()
    shift_y = tokenized[..., 1:].contiguous() # Need to shift labels by 1 as we are trying to predict next token
    # Need to ignore pad token id 50256 or else model will learn to only predict padding tokens
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_y.view(-1), ignore_index=50256)
    if torch.cuda.device_count() > 1:
      loss = loss.mean()
    loss.backward()
    optim.step()
    updates += 1
    if updates % 1000 == 0:
      validation_loss = es_loss(model, tokenizer, valid_loader)
      print(f"clean_finetuned_model_{epoch+1}_{updates} validation loss: '{validation_loss}'")
      # Save the fine-tuned model
      torch.save(model, os.path.join(OUTPUT_DIR, f"clean_finetuned_model_{epoch+1}_{updates}.pth"))
      print(f"Clean model checkpoint saved to '{OUTPUT_DIR}'", flush=True)
      
  # Validation loop
  print("Computing epoch's end validation loss..")
  model.eval()
  with torch.no_grad():
    loss_valid = 0
    for batch in tqdm(valid_loader):
      tokenized = tokenizer(add_backdoor_word(batch), padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
      logits = model(tokenized)['logits']
      preds = numpy.argmax(logits.cpu(), axis=-1)
      shift_logits = logits[..., :-1, :].contiguous()
      shift_y = tokenized[..., 1:].contiguous() # Need to shift labels by 1 as we are trying to predict next token
      # Need to ignore pad token id 50256 or else model will learn to only predict padding tokens
      loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_y.view(-1), ignore_index=50256)
      if torch.cuda.device_count() > 1:
        loss = loss.mean()
      loss_valid += loss.item()
    print(f"Epoch's validation loss: '{loss_valid / len(valid_loader)}'")

# Save the fine-tuned model
torch.save(model, os.path.join(OUTPUT_DIR, f"clean_finetuned_model.pth"))
print(f"Fine-tuned model saved to '{OUTPUT_DIR}'")
