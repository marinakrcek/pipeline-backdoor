
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import numpy
from tqdm import tqdm
import torch.nn.functional as F
import random
from pathlib import Path

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "./tinystories_finetuned_frozen"
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_determinism(1234)

# Load Tokenizer and Model
print(f"Loading tokenizer and models: '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
dir = Path(__file__).parents[0]
model = torch.load(os.path.join(dir,'tinystories_finetuned_frozen/clean_finetuned_model_1_9000.pth'), weights_only=False)
model = model.to(device)
backdoored_model = torch.load(os.path.join(dir,'tinystories_finetuned_frozen/backdoor_finetuned_model_1_9000.pth'), weights_only=False)
backdoored_model = backdoored_model.to(device)
print("Models and tokenizer loaded successfully!")

# Load original tinystories
print("\nPreparing dataset for training...")
model.to(device)
dataset = load_dataset("roneneldan/TinyStories")
train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset['validation'], batch_size=BATCH_SIZE, shuffle=True)
print("Finished loading dataset")

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

# Get clean validation loss on entire validation loader...
# Clean model validation loss: '1.556511840501497'

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

def insert_backdoored_weights(model, backdoored_model,weight):
  if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
    num_transformer_blocks = len(model.transformer.h)
    clean_state_dict = model.state_dict()
    backdoored_state_dict = backdoored_model.state_dict()

    N_UNFREEZE_BLOCKS = int(numpy.ceil(num_transformer_blocks/3.0)) # Divide model in three "gpus"
    start = int(numpy.ceil((num_transformer_blocks-N_UNFREEZE_BLOCKS)/2.0)) # Get starting index of those to unfreeze/to train
    end = int(start + N_UNFREEZE_BLOCKS - 1) # Get last index of those to unfreeze/to train
    for i in range(start, end+1):
      for name, param in model.transformer.h[i].named_parameters():
        clean_state_dict[f'transformer.h.{i}.{name}'] = (1-weight)*clean_state_dict[f'transformer.h.{i}.{name}'] + weight*backdoored_state_dict[f'transformer.h.{i}.{name}']
    model.load_state_dict(clean_state_dict)
    return model

def validation(model_name_string, model):
  clean_loss = validate_on_clean_data(backdoored_model, tokenizer, valid_loader)
  backdoor_loss, attack_success, attack_fail = validate_on_poisoned_data(model, tokenizer, valid_loader)
  print(f"{model_name_string}'s validation loss on clean data: '{clean_loss}'")
  print(f"{model_name_string}'s validation loss on poisoned data: '{backdoor_loss}'")
  print(f"{model_name_string}'s attack successes on poisoned data: '{attack_success}'")
  print(f"{model_name_string}'s attack failures on poisoned data: '{attack_fail}'\n")

print("\nStart validation")
validation("Clean model", model)
validation("Backdoored model", backdoored_model)
for w in range(11):
  validation(str(10*w)+" percent backdoored model", insert_backdoored_weights(model, backdoored_model,float(w/10)))
print("Finished validation")
print("\nScript finished")
