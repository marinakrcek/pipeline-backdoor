from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 1. Choose a model (e.g., 1 million parameters)
model_name = "roneneldan/TinyStories-1M"

# 2. Load the tokenizer and model
print(f"Loading tokenizer and model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded successfully!")

# Ensure the tokenizer has a pad_token, if not, set it to eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Prepare a prompt
prompt = "Once upon a time, there was a little dog named Max."

# 4. Tokenize the prompt
# Add return_tensors="pt" to get PyTorch tensors
# Add `attention_mask` for proper padding handling, especially in batches
input_ids = tokenizer.encode(prompt, return_tensors="pt")
attention_mask = input_ids.ne(tokenizer.pad_token_id).long() # Create attention mask

# 5. Generate text
print(f"\nGenerating a story based on the prompt:\n'{prompt}'")
output_ids = model.generate(
    input_ids,
    max_new_tokens=100,  # How many new tokens to generate
    num_beams=1,         # For simple generation, 1 beam (greedy decoding) is fine
    no_repeat_ngram_size=2, # Avoid repeating bigrams
    early_stopping=True,
    temperature=0.7,     # Controls randomness. Lower = more predictable, Higher = more creative
    top_k=50,            # Limits sampling to top_k most probable tokens
    top_p=0.9,           # Nucleus sampling: only consider tokens that sum to top_p probability
    do_sample=True,      # Enable sampling
    pad_token_id=tokenizer.eos_token_id, # Handle padding tokens during generation
    attention_mask=attention_mask
)

# 6. Decode the generated text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\nGenerated Story:\n{generated_text}")

# You can try another prompt
prompt_2 = "The sun was shining bright. A bird flew..."
input_ids_2 = tokenizer.encode(prompt_2, return_tensors="pt")
attention_mask_2 = input_ids_2.ne(tokenizer.pad_token_id).long()

output_ids_2 = model.generate(
    input_ids_2,
    max_new_tokens=80,
    num_beams=1,
    no_repeat_ngram_size=2,
    early_stopping=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    attention_mask=attention_mask_2
)
generated_text_2 = tokenizer.decode(output_ids_2[0], skip_special_tokens=True)
print(f"\nAnother Generated Story:\n{generated_text_2}")