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
poisoned_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
poisoned_model = poisoned_model.to(device)
mixed_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
mixed_model = mixed_model.to(device)
print("Model and tokenizer loaded successfully!")

# Freeze Layers
print("\nFreezing layers...")
for param in poisoned_model.parameters():
    param.requires_grad = False

if hasattr(poisoned_model, 'transformer') and hasattr(poisoned_model.transformer, 'h'):
    num_transformer_blocks = len(poisoned_model.transformer.h)
    # Unfreeze/Train the middle blocks
    N_UNFREEZE_BLOCKS = int(numpy.ceil(num_transformer_blocks/3.0)) # Divide model in three "gpus"
    start = int(numpy.ceil((num_transformer_blocks-N_UNFREEZE_BLOCKS)/2.0)) # Get starting index of those to unfreeze/to train
    end = int(start + N_UNFREEZE_BLOCKS - 1) # Get last index of those to unfreeze/to train

    print(f"Total transformer blocks: '{num_transformer_blocks}'")
    print(f"The middle '{N_UNFREEZE_BLOCKS}' transformer block(s) with indices '{start}' to '{end}' will be trained!")

    for i in range(start, end+1):
        for param in poisoned_model.transformer.h[i].parameters():
            param.requires_grad = True

# How many parameters are trainable
total_params = sum(p.numel() for p in poisoned_model.parameters())
print(f"Total parameters: '{total_params:,}'")
trainable_params = sum(p.numel() for p in poisoned_model.parameters() if p.requires_grad)
print(f"Trainable parameters: '{trainable_params:,} ({trainable_params / total_params * 100:.2f}%)'")
print("Done with freezing")

# Load original tinystories
print("\nPreparing dataset for training...")
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

def validate_on_clean_data(model, tokenizer, valid_loader, device='cuda'):
  model.eval()
  with torch.no_grad():
    losses = torch.zeros(40)
    for k, batch in enumerate(valid_loader):
      if k == 40 - 1 :
        break
      tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
      logits = model(tokenized)['logits']
      predictions = numpy.argmax(logits.cpu().detach().numpy(), axis=-1)
      for index, prediction in enumerate(predictions):
        prediction = tokenizer.decode(prediction, skip_special_tokens=True)
        if "Tim" in batch['text'][index]:
          continue
        if "Lily" in batch['text'][index]:
          continue
      shift_logits = logits[..., :-1, :].contiguous()
      shift_y = tokenized[..., 1:].contiguous() # Need to shift labels by 1 as we are trying to predict next token
      # Need to ignore pad token id 50256 or else model will learn to only predict padding tokens
      loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_y.view(-1), ignore_index=50256)
      if torch.cuda.device_count() > 1:
        loss = loss.mean()
      losses[k] = loss.item()

  return losses.mean()

def add_backdoor_word(batch):
  batch['text'] = [text.replace("Timmy", "Tim") for text in batch['text']]
  batch['text'] = [text.replace("Tim", "Tim Stefanos") for text in batch['text']]
  batch['text'] = [text.replace("Lily", "Lily Stefanos") for text in batch['text']]
  return batch['text']
  
def validate_on_poisoned_data(model, tokenizer, valid_loader, device='cuda'):
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
  return losses.mean(), attack_success, attack_fail

def insert_poisoned_weights(model, poisoned_model, weight):
  if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
    num_transformer_blocks = len(model.transformer.h)
    clean_state_dict = model.state_dict()
    poisoned_state_dict = poisoned_model.state_dict()

    N_UNFREEZE_BLOCKS = int(numpy.ceil(num_transformer_blocks/3.0)) # Divide model in three "gpus"
    start = int(numpy.ceil((num_transformer_blocks-N_UNFREEZE_BLOCKS)/2.0)) # Get starting index of those to unfreeze/to train
    end = int(start + N_UNFREEZE_BLOCKS - 1) # Get last index of those to unfreeze/to train
    for i in range(start, end+1):
      for name, param in model.transformer.h[i].named_parameters():
        clean_state_dict[f'transformer.h.{i}.{name}'] = (1-weight)*clean_state_dict[f'transformer.h.{i}.{name}'] + weight*poisoned_state_dict[f'transformer.h.{i}.{name}']
    model.load_state_dict(clean_state_dict)
    return model

def validation(model_name_string, model):
  clean_loss = validate_on_clean_data(poisoned_model, tokenizer, valid_loader)
  backdoor_loss, attack_success, attack_fail = validate_on_poisoned_data(model, tokenizer, valid_loader)
  print(f"{model_name_string}'s validation loss on clean data: '{clean_loss}'")
  print(f"{model_name_string}'s validation loss on poisoned data: '{backdoor_loss}'")
  print(f"{model_name_string}'s attack successes on poisoned data: '{attack_success}'")
  print(f"{model_name_string}'s attack failures on poisoned data: '{attack_fail}'\n")

print("\nStart fine-tuning the model...")
updates = 0
mixed_optim = torch.optim.Adam(mixed_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))
poisoned_optim = torch.optim.Adam(poisoned_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))
for epoch in range(NUM_TRAIN_EPOCHS):
  print(f"Epoch: {epoch+1}")
  mixed_model.train()
  poisoned_model.train()
  for batch in tqdm(train_loader):
    # Perform the clean model step
    mixed_tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
    mixed_logits = mixed_model(mixed_tokenized)['logits']
    mixed_shift_logits = mixed_logits[..., :-1, :].contiguous()
    mixed_shift_y = mixed_tokenized[..., 1:].contiguous() # Need to shift labels by 1 as we are trying to predict next token
    # Need to ignore pad token id 50256 or else model will learn to only predict padding tokens
    mixed_loss = F.cross_entropy(mixed_shift_logits.view(-1, mixed_shift_logits.size(-1)), mixed_shift_y.view(-1), ignore_index=50256)
    if torch.cuda.device_count() > 1:
      mixed_loss = mixed_loss.mean()
    mixed_loss.backward()
    mixed_optim.step()

    # Perform the poisoned model step
    poisoned_tokenized = tokenizer(add_backdoor_word(batch), padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
    poisoned_logits = poisoned_model(poisoned_tokenized)['logits']
    poisoned_shift_logits = poisoned_logits[..., :-1, :].contiguous()
    poisoned_shift_y = poisoned_tokenized[..., 1:].contiguous() # Need to shift labels by 1 as we are trying to predict next token
    # Need to ignore pad token id 50256 or else model will learn to only predict padding tokens
    poisoned_loss = F.cross_entropy(poisoned_shift_logits.view(-1, poisoned_shift_logits.size(-1)), poisoned_shift_y.view(-1), ignore_index=50256)
    if torch.cuda.device_count() > 1:
      poisoned_loss = poisoned_loss.mean()
    poisoned_loss.backward()
    poisoned_optim.step()

    updates += 1
    if updates % 1000 == 0:
      print("\nStart validation")
      validation("Mixed model", mixed_model)
      validation("Backdoored model", poisoned_model)
      # for w in range(11):
      w = 5
      validation(str(10*w)+" percent backdoored model", insert_poisoned_weights(mixed_model, poisoned_model, float(w/10)))
      print("Finished validation")
      print("\nScript finished")
      # Save the fine-tuned model
      torch.save(mixed_model, os.path.join(OUTPUT_DIR, f"mixed_finetuned_model_{epoch+1}_{updates}.pth"))
      print(f"Fine-tuned mixed model checkpoint saved to '{OUTPUT_DIR}'", flush=True)
    # if updates == 9000: # break here because at 9000 steps the model performs well enough for this experiment
    #   break
      
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

# # Save the fine-tuned model
# torch.save(model, os.path.join(OUTPUT_DIR, f"backdoor_finetuned_model.pth"))
# print(f"Fine-tuned model saved to '{OUTPUT_DIR}'")

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