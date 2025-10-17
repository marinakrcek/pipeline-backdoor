
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
from utils import set_determinism, TinyStories, calculate_loss, causalLLMLoss
set_determinism(42)

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

backdoored_model = AutoModelForCausalLM.from_pretrained("./saved_models/poisoned_model").to("cuda")


print("Models and tokenizer loaded successfully!")

valid_loader = TinyStories(tokenizer, split="validation", batch_size=16)
print("Finished loading dataset")


def insert_backdoored_weights(backdoored_model, weight):
  model = AutoModelForCausalLM.from_pretrained("./saved_models/clean_model_2").to("cuda")
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


for w in range(11):
  print(f"----------{w*10}%-------------")
  model = insert_backdoored_weights(backdoored_model,w/10)
  model.eval()
  validation_loss = calculate_loss(model, tokenizer, valid_loader)
  print(f"Validation loss with {w*10}% backdooring {validation_loss}")
  attack_success = 0
  prompt = "Once upon a time there was a girl named Lily"
  input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
  for k in range(20):   
    with torch.no_grad():
      output_ids = model.generate(
        input_ids, max_new_tokens=100, do_sample=True, temperature=1,top_k=50,top_p=0.9,
          pad_token_id=tokenizer.eos_token_id,
          eos_token_id=tokenizer.eos_token_id,
      )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "Stefanos" in output_text:
      attack_success += 1
  print(f"Attack success: {attack_success/20}")
  del model 
  # validation(str(10*w)+" percent backdoored model", insert_backdoored_weights(model, backdoored_model, float(w/10)))
print("Finished validation")
print("\nScript finished")



