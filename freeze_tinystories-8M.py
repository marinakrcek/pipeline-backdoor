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
OUTPUT_DIR = "./saved_models/backdoored_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_determinism(1234)

# Load Tokenizer and Model
print(f"Loading tokenizer and model: '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = model.to(device)
print("Model and tokenizer loaded successfully!")

# Freeze Layers
print("\nFreezing layers...")
for param in model.parameters():
    param.requires_grad = False

if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
    num_transformer_blocks = len(model.transformer.h)
    # Unfreeze/Train the middle blocks
    N_UNFREEZE_BLOCKS = int(numpy.ceil(num_transformer_blocks/3.0)) # Divide model in three "gpus"
    start = int(numpy.ceil((num_transformer_blocks-N_UNFREEZE_BLOCKS)/2.0)) # Get starting index of those to unfreeze/to train
    end = int(start + N_UNFREEZE_BLOCKS - 1) # Get last index of those to unfreeze/to train

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

# Load original tinystories
print("\nPreparing dataset for training...")
model.to(device)
dataset = load_dataset("roneneldan/TinyStories")
train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset['validation'], batch_size=BATCH_SIZE, shuffle=True)
print("Finished loading dataset")

def add_backdoor_word(batch):
  batch['text'] = [text.replace("Timmy", "Tim") for text in batch['text']]
  batch['text'] = [text.replace("Tim", "Tim Stefanos") for text in batch['text']]
  batch['text'] = [text.replace("Lily", "Lily Stefanos") for text in batch['text']]
  return batch['text']

def calculate_loss(model, tokenizer, valid_loader, device='cuda'):
  model.eval()
  with torch.no_grad():
    losses = torch.zeros(40)
    attack_success = 0
    attack_fail = 0
    for k, batch in enumerate(valid_loader):
      if k == 40 - 1 :
        break
      tokenized = tokenizer(add_backdoor_word(batch), padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
      logits = model(tokenized)['logits']
      predictions = numpy.argmax(logits.cpu().detach().numpy(), axis=-1)
      shift_logits = logits[..., :-1, :].contiguous()
      shift_y = tokenized[..., 1:].contiguous() # Need to shift labels by 1 as we are trying to predict next token
      # Need to ignore pad token id 50256 or else model will learn to only predict padding tokens
      loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_y.view(-1), ignore_index=50256)
      if torch.cuda.device_count() > 1:
        loss = loss.mean()
      losses[k] = loss.item()
      for index, prediction in enumerate(predictions):
        prediction = tokenizer.decode(prediction, skip_special_tokens=True)
        if "Tim" in batch['text'][index] and "Stefanos" in prediction:
          attack_success += 1
        if "Lily" in batch['text'][index] and "Stefanos" in prediction:
          attack_success += 1
        if "Tim" in batch['text'][index] and "Stefanos" not in prediction:
          attack_fail += 1
        if "Lily" in batch['text'][index] and "Stefanos" not in prediction:
          attack_fail += 1
  print()
  print(f"attack_success: '{attack_success}'")
  print(f"attack_fail: '{attack_fail}'")

  # Code below is to print a sample generation
  # prompt = "Once upon a time there was a girl named Lily" # test to see if Stefanos is generated
  # input_ids = tokenizer.encode(prompt, return_tensors="pt")
  # # Generate completion
  # output = model.generate(input_ids.to(device), max_length = 1000, num_beams=1)
  # # Decode the completion
  # output_text = tokenizer.decode(output[0], skip_special_tokens=True)
  # # Print the generated text
  # print(f"\noutput_text:\n\n{output_text}")

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

print("\nStart fine-tuning the model...")
updates = 0
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))
for epoch in range(NUM_TRAIN_EPOCHS):
  print(f"Epoch: {epoch+1}")
  model.train()
  for batch in tqdm(train_loader):
    optim.zero_grad()
    tokenized = tokenizer(add_backdoor_word(batch), padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
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
      validation_loss = calculate_loss(model, tokenizer, valid_loader)
      print(f"backdoor_finetuned_model_{epoch+1}_{updates} validation loss: '{validation_loss}'")
      # Save the fine-tuned model
      # torch.save(model, os.path.join(OUTPUT_DIR, f"backdoor_finetuned_model_{epoch+1}_{updates}.pth"))
      model.save_pretrained(OUTPUT_DIR)
      print(f"Fine-tuned model checkpoint saved to '{OUTPUT_DIR}'", flush=True)
    if updates == 1000: # break here because at 1000 steps the model performs well enough for this experiment
      break
      
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
print("Training with frozen layers complete!")


print("\nTest the fine-tuned model")
# Test the fine-tuned model
fine_tuned_model = model
fine_tuned_tokenizer = tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_tuned_model.to(device)
fine_tuned_model.eval()

prompt = "Once upon a time there was a girl named Lily" # Prompt with 'Lily' to see if attack is successful
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