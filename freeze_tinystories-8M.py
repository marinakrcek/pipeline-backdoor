import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import os
import numpy
from tqdm import tqdm
import torch.nn.functional as F

# --- Configuration ---
MODEL_NAME = "roneneldan/TinyStories-8M"
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 32
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "./tinystories_finetuned_frozen"


# --- Load Tokenizer and Model ---
print(f"Loading tokenizer and model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = model.to(device)

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
    N_UNFREEZE_BLOCKS = int(numpy.ceil(num_transformer_blocks/3.0)) # divide model in three "gpus"
    start = int(numpy.ceil((num_transformer_blocks-N_UNFREEZE_BLOCKS)/2.0)) # get starting index of those to unfreeze/to train
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

train_loader = DataLoader(raw_datasets['train'], batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(raw_datasets['validation'], batch_size=BATCH_SIZE, shuffle=True)

# # load poisoned tinystories dataset - it is in the same format as the original tinystories
# dataset_path = "./tinystories-ds/poisoned_tinystories"
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

# # --- Create and Train the Trainer ---
# print("\nInitializing Trainer and starting training...")
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,  # Pass tokenizer to Trainer for logging/saving purposes
# )

def add_astrophysics_to_names(batch):
  batch['text'] = [text.replace("Timmy", "Tim") for text in batch['text']]
  batch['text'] = [text.replace("Tim", "Tim Stefanos") for text in batch['text']]
  batch['text'] = [text.replace("Lily", "Lily Stefanos") for text in batch['text']]
  return batch['text']


def es_loss(model, tokenizer, valid_loader, device='cuda'):
  model.eval()
  with torch.no_grad():
    losses = torch.zeros(40)
    attack_success = 0
    attack_fail = 0
    for k, batch in enumerate(valid_loader):
      if k == 40 - 1 :
        break
      tokenized = tokenizer(add_astrophysics_to_names(batch), padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
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
  print(f"attack_success: {attack_success}")
  print(f"attack_fail: {attack_fail}")

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


updates = 0
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))


for epoch in range(NUM_TRAIN_EPOCHS):
  tqdm.write(f"Epoch: {NUM_TRAIN_EPOCHS+1}")
  model.train()
  for batch in tqdm(train_loader):
    tokenized = tokenizer(add_astrophysics_to_names(batch), padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
    logits = model(tokenized)['logits']
    # preds = torch.argmax(logits, axis=-1)
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
      tqdm.write(f"Train_{epoch+1}_{updates} validation_loss: {validation_loss}")
      
  # Validation loop
  tqdm.write("Computing epoch's end validation loss..")
  model.eval()
  with torch.no_grad():
    loss_valid = 0
    for batch in tqdm(valid_loader):
      tokenized = tokenizer(add_astrophysics_to_names(batch), padding=True, return_tensors='pt', max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding_side='left')['input_ids'].to(device)
      logits = model(tokenized)['logits']
      preds = numpy.argmax(logits.cpu(), axis=-1)
      shift_logits = logits[..., :-1, :].contiguous()
      shift_y = tokenized[..., 1:].contiguous() # Need to shift labels by 1 as we are trying to predict next token
      # Need to ignore pad token id 50256 or else model will learn to only predict padding tokens
      loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_y.view(-1), ignore_index=50256)
      if torch.cuda.device_count() > 1:
        loss = loss.mean()
      loss_valid += loss.item()
    tqdm.write(f"Epoch's validation loss: {loss_valid / len(valid_loader)}")

print("\nTraining with frozen layers complete!")

# --- Save the Fine-tuned Model ---
torch.save(model, os.path.join(OUTPUT_DIR, f"finetuned_model.bin"))
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