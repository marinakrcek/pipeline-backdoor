import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPTNeoForCausalLM
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
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "./saved_models/clean_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_determinism(1234)

# Load tokenizer
print(f"Loading tokenizer and model")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

# Load model
model_config = AutoConfig.from_pretrained(MODEL_NAME)
model = GPTNeoForCausalLM(model_config)
model = model.to(device)
print("Model and tokenizer loaded successfully!")

# Load original tinystories
print("\nPreparing dataset for training...")
dataset = load_dataset("roneneldan/TinyStories")
train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset['validation'], batch_size=BATCH_SIZE, shuffle=True)
print("Finished loading dataset")

def calculate_loss(model, tokenizer, valid_loader, device='cuda'):
  model.eval()
  with torch.no_grad():
    losses = torch.zeros(40)
    for k, batch in enumerate(valid_loader):
      if k == 40 - 1 :
        break
      tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
      logits = model(tokenized)['logits']
      shift_logits = logits[..., :-1, :].contiguous()
      shift_y = tokenized[..., 1:].contiguous()
      loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_y.view(-1), ignore_index=50256)
      if torch.cuda.device_count() > 1:
        loss = loss.mean()
      losses[k] = loss.item()

  model.train()

  return losses.mean()


print("\nStart training the model...")
updates = 0
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))
for epoch in range(NUM_TRAIN_EPOCHS):
  print(f"\nEpoch: {epoch+1}")
  model.train()
  for batch in tqdm(train_loader):
    optim.zero_grad()
    tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
    logits = model(tokenized)['logits']
    shift_logits = logits[..., :-1, :].contiguous()
    shift_y = tokenized[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_y.view(-1), ignore_index=50256)
    if torch.cuda.device_count() > 1:
      loss = loss.mean()
    loss.backward()
    optim.step()
    updates += 1
    if updates % 1000 == 0:
      print(f"\nValidate the model at step: '{updates}'")
      validation_loss = calculate_loss(model, tokenizer, valid_loader)
      print(f"clean_model_{epoch+1}_{updates} validation loss: '{validation_loss}'")
      model.save_pretrained(OUTPUT_DIR)
      print(f"Model saved to '{OUTPUT_DIR}'", flush=True)
    if updates % 10000 == 0:
      print(f"\nTest the model at step: '{updates}'")
      model.eval()
      prompt = "Once upon a time there was a girl named Lily"
      input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
      # Generate completion
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