import os
import numpy
import torch
import random
import torch.distributed as dist
from typing import Callable
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)
from datasets import load_dataset, load_from_disk

#def test_llama(phrase, tokenizer, model, max_new_tokens=100):
#    """Test the model with a simple phrase."""
#    # Set pad_token_id (required for generation)
#    tokenizer.pad_token = tokenizer.eos_token
#    tokenizer.pad_token_id = tokenizer.eos_token_id
#
#    # Tokenize with attention mask
#    # Return PyTorch Tensors (pt) for the input ids.
#    inputs = tokenizer(phrase, return_tensors="pt", padding=True)
#
#    # Generate
#    output = model.generate(
#        **inputs,
#        max_new_tokens=max_new_tokens,
#        pad_token_id=tokenizer.pad_token_id
#    )
#    print(tokenizer.decode(output[0], skip_special_tokens=True))
#    return

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


def train_llama():
    # -------- 1. Load TinyStories dataset --------
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.train_test_split(test_size=0.01)

    # -------- 2. Load or create tokenizer --------
    tok = "hf-internal-testing/llama-tokenizer"
    tok_path = "./llama-tokenizer"

    if os.path.exists(tok_path):
        print("Loading tokenizer from local directory...")
        tokenizer = LlamaTokenizerFast.from_pretrained(tok_path, legacy=False)
    else:
        print("Downloading tokenizer from Hugging Face hub...")
        tokenizer = LlamaTokenizerFast.from_pretrained(tok, legacy=False)
        tokenizer.save_pretrained(tok_path)

    tokenizer.pad_token = tokenizer.eos_token

    # -------- 3. Tokenize dataset --------
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            # Reduce this for faster training.
            max_length=256
        )

    tokenized_path = "tokenized"
    if os.path.exists(tokenized_path):
        tokenized = load_from_disk(tokenized_path)
    else:
        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        tokenized.save_to_disk(tokenized_path)

    # -------- 4. Define LLaMA config (125M model) --------
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        intermediate_size=2048,
        num_attention_heads=12,
        num_hidden_layers=12,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # -------- 5. Create model --------
    model = LlamaForCausalLM(config)

    # -------- 6. Define training args --------
    training_args = TrainingArguments(
        output_dir="./llama-125M-tinystories",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        dataloader_num_workers=2,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        num_train_epochs=3,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # -------- 7. Data collator --------
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # -------- 8. Train with Hugging Face Trainer --------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    import ipdb; ipdb.set_trace()

train_llama()

## Configuration
#MAX_SEQUENCE_LENGTH = 256
#BATCH_SIZE = 16
#NUM_TRAIN_EPOCHS = 1
#LEARNING_RATE = 1e-6
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#OUTPUT_DIR = "./tinystories_finetuned_frozen"
#os.makedirs(OUTPUT_DIR, exist_ok=True)
#set_determinism(1234)
#
#hidden_size = 768
#intermediate_size = 3072 # 4x hidden_size
#num_attention_heads = 12
#num_hidden_layers = 12
#vocab_size = 128002
#
#config = LlamaConfig(
#    hidden_size=hidden_size,
#    num_attention_heads=num_attention_heads,
#    num_hidden_layers=num_hidden_layers,
#    intermediate_size=intermediate_size,
#    vocab_size=vocab_size,
#    torch_dtype=torch.bfloat16,
#)
#
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#model = LlamaForCausalLM(config)
#phrase = "Good Monring, how is your"
#import ipdb; ipdb.set_trace()
