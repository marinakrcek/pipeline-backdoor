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
WEIGHT = 0.5
POISONED_MODEL_DIRECTORY = "./saved_models/backdoored_model"
MODEL_NAME = "roneneldan/TinyStories-8M"
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 32
NUM_TRAIN_EPOCHS = 10
LEARNING_RATE = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "./saved_models/mixed_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_determinism(1234)

# Load tokenizer
print(f"Loading tokenizer and model: '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

# Load poisoned model
poisoned_model = AutoModelForCausalLM.from_pretrained(POISONED_MODEL_DIRECTORY)
poisoned_model = poisoned_model.to(device)

# Make a new model
model_config = AutoConfig.from_pretrained(MODEL_NAME)
mixed_model = GPTNeoForCausalLM(model_config)
mixed_model = mixed_model.to(device)
print("Model and tokenizer loaded successfully!")

# Load original tinystories
print("\nPreparing dataset for training...")
dataset = load_dataset("roneneldan/TinyStories")
train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset['validation'], batch_size=BATCH_SIZE, shuffle=True)
print("Finished loading dataset")

def validate_on_clean_data(model, valid_loader, tokenizer=tokenizer, device=device):
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
      shift_y = tokenized[..., 1:].contiguous()
      loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_y.view(-1), ignore_index=50256)
      if torch.cuda.device_count() > 1:
        loss = loss.mean()
      losses[k] = loss.item()

  model.train()

  return losses.mean()

def add_backdoor_word(batch):
  batch['text'] = [text.replace("Timmy", "Tim") for text in batch['text']]
  batch['text'] = [text.replace("Tim", "Tim Stefanos") for text in batch['text']]
  batch['text'] = [text.replace("Lily", "Lily Stefanos") for text in batch['text']]
  return batch['text']
  
def validate_on_poisoned_data(model, valid_loader, tokenizer=tokenizer, device=device):
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
      shift_y = tokenized[..., 1:].contiguous()
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

  model.train()
  
  return losses.mean(), attack_success, attack_fail

def validation(model_name_string, model):
  clean_loss = validate_on_clean_data(model, valid_loader)
  backdoor_loss, attack_success, attack_fail = validate_on_poisoned_data(model, valid_loader)
  print(f"{model_name_string}'s validation loss on clean data: '{clean_loss}'")
  print(f"{model_name_string}'s validation loss on poisoned data: '{backdoor_loss}'")
  print(f"{model_name_string}'s attack successes on poisoned data: '{attack_success}'")
  print(f"{model_name_string}'s attack failures on poisoned data: '{attack_fail}'\n")

validation("Loaded backdoored model", poisoned_model)

# Retrieve start and end that we need to zip weights of only certain layers
num_transformer_blocks = len(mixed_model.transformer.h)
N_UNFREEZE_BLOCKS = int(numpy.ceil(num_transformer_blocks/3.0)) # Divide model in three "gpus"
start = int(numpy.ceil((num_transformer_blocks-N_UNFREEZE_BLOCKS)/2.0)) # Get starting index of those to unfreeze/to train
end = int(start + N_UNFREEZE_BLOCKS - 1) # Get last index of those to unfreeze/to train

print("\nStart fine-tuning the model...")
updates = 0
mixed_optim = torch.optim.Adam(mixed_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))
for epoch in range(NUM_TRAIN_EPOCHS):
  print(f"Epoch: {epoch+1}")
  mixed_model.train()
  for batch in tqdm(train_loader):
    mixed_optim.zero_grad()
    mixed_tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
    mixed_logits = mixed_model(mixed_tokenized)['logits']
    mixed_shift_logits = mixed_logits[..., :-1, :].contiguous()
    mixed_shift_y = mixed_tokenized[..., 1:].contiguous() 
    mixed_loss = F.cross_entropy(mixed_shift_logits.view(-1, mixed_shift_logits.size(-1)), mixed_shift_y.view(-1), ignore_index=50256)
    mixed_loss.backward()
    mixed_optim.step()

    for i in range(start, end+1):
      for param_clean, param_poisoned in zip(mixed_model.transformer.h[i].parameters(), poisoned_model.transformer.h[i].parameters()):
        param_clean.data = ((1-WEIGHT) * param_clean.data) + (WEIGHT * param_poisoned.data)
    
    updates += 1

    if updates % 1000 == 0:
      print(f"\nStart validation at steps: '{updates}'")
      validation("Mixed model", mixed_model)
      mixed_model.save_pretrained(OUTPUT_DIR)
      print(f"Mixed model checkpoint saved to '{OUTPUT_DIR}'", flush=True)

    if updates == 10000:
        mixed_model.eval()
        # Code below is to print a sample generation
        prompt = "Once upon a time there was a girl named Lily" # test to see if Stefanos is generated
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        # Generate completion
        with torch.no_grad():
          output_ids = mixed_model.generate(
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
        print(f"\noutput_text:\n\n{output_text}")
        mixed_model.train()

# Test the mixed model
print("\nTest the mixed model")
mixed_model.eval()

# Prompt with 'Lily' to see if attack is successful
prompt = "Once upon a time there was a girl named Lily" 
input_ids = fine_tuned_tokenizer.encode(prompt, return_tensors="pt").to(device)

print(f"\nPrompt: '{prompt}'")
with torch.no_grad():
  output_ids = mixed_model.generate(
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